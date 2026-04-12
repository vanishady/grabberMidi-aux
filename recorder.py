"""
recorder.py — Synchronized MIDI + Audio Recorder

Records MIDI events and audio simultaneously with coherent internet timestamps.
Timestamps are computed using time.perf_counter() relative to a session anchor,
with an optional single NTP query at session start for UTC correction.

Usage:
    python recorder.py

Requirements:
    pip install python-rtmidi sounddevice soundfile ntplib
"""
from __future__ import annotations

import csv
import datetime
import os
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    import ntplib
except ImportError:
    ntplib = None  # type: ignore

import rtmidi
import sounddevice as sd
import soundfile as sf
import numpy as np


# ---------------------------------------------------------------------------
# MIDI note number → name  (C4 = MIDI 60)
# ---------------------------------------------------------------------------
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_note_to_name(note: int) -> str:
    """Convert a MIDI note number (0-127) to a human-readable name (e.g. 'C4')."""
    name = _NOTE_NAMES[note % 12]
    octave = (note // 12) - 1
    return f"{name}{octave}"


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def get_ntp_offset(server: str = "pool.ntp.org") -> float:
    """
    Query an NTP server ONCE and return the offset in seconds:
        corrected_utc = datetime.utcnow() + timedelta(seconds=offset)
    Raises RuntimeError if ntplib is not installed, or NTPException on failure.
    """
    if ntplib is None:
        raise RuntimeError("ntplib is not installed. Run: pip install ntplib")
    c = ntplib.NTPClient()
    response = c.request(server, version=3, timeout=5)
    return response.offset


def make_iso_timestamp(
    perf_offset: float,
    session_start_utc: datetime.datetime,
) -> str:
    """
    Build an ISO 8601 / UTC timestamp string for a single event.

    :param perf_offset:       time.perf_counter() at event  -  session_start_perf
    :param session_start_utc: UTC datetime of session start (already NTP-corrected if requested)
    :returns:                 e.g. '2026-04-12T14:23:45.123456Z'
    """
    event_utc = session_start_utc + datetime.timedelta(seconds=perf_offset)
    return event_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ---------------------------------------------------------------------------
# MIDI status byte parsing
# ---------------------------------------------------------------------------

_MIDI_TYPE_MAP: dict[int, str] = {
    0x80: "note_off",
    0x90: "note_on",
    0xA0: "aftertouch",
    0xB0: "control_change",
    0xC0: "program_change",
    0xD0: "channel_pressure",
    0xE0: "pitch_bend",
    0xF0: "sysex",
}


def parse_midi_status(status: int) -> tuple[str, int]:
    """Return (event_type_str, channel_0based) from a MIDI status byte."""
    msg_type = status & 0xF0
    channel = status & 0x0F
    event_type = _MIDI_TYPE_MAP.get(msg_type, f"unknown_0x{msg_type:02X}")
    return event_type, channel


# ---------------------------------------------------------------------------
# MidiRecorder
# ---------------------------------------------------------------------------

class MidiRecorder:
    """
    Opens a MIDI input port via python-rtmidi and accumulates raw events.

    The RtMidi callback runs in a dedicated C++ thread; the only work done
    there is appending (perf_counter, message_bytes) to a list under a lock,
    so real-time performance is not impacted.
    """

    def __init__(self) -> None:
        self._midi_in: rtmidi.MidiIn | None = None
        self.raw_events: list[tuple[float, list[int]]] = []
        self._lock = threading.Lock()
        self._session_start_perf: float = 0.0

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def get_port_names() -> list[str]:
        tmp = rtmidi.MidiIn()
        names = [tmp.get_port_name(i) for i in range(tmp.get_port_count())]
        del tmp
        return names

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start(self, port_index: int, session_start_perf: float) -> None:
        with self._lock:
            self.raw_events.clear()
        self._session_start_perf = session_start_perf
        self._midi_in = rtmidi.MidiIn()
        # Keep sysex; ignore clock & active-sense (high-frequency, not needed)
        self._midi_in.ignore_types(sysex=False, timing=True, active_sense=True)
        self._midi_in.open_port(port_index)
        self._midi_in.set_callback(self._callback)

    def _callback(self, event: tuple, data: object = None) -> None:
        # event = ([status, d1, d2], delta_seconds_since_last_event)
        message, _delta = event
        t = time.perf_counter()
        with self._lock:
            self.raw_events.append((t, list(message)))

    def stop(self) -> None:
        if self._midi_in is not None:
            self._midi_in.close_port()
            self._midi_in = None

    def get_events(self) -> list[tuple[float, list[int]]]:
        with self._lock:
            return list(self.raw_events)


# ---------------------------------------------------------------------------
# AudioRecorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """
    Opens a sounddevice InputStream and accumulates int16 chunks.

    The PortAudio callback does only one thing: queue.put(indata.copy()).
    All heavy work (concatenation, WAV writing) happens after stop().
    """

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.sample_rate: int = 44100
        self.channels: int = 1

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def get_device_list() -> list[dict]:
        """Return list of dicts with 'index' and 'name' for input-capable devices."""
        result = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                result.append({"index": i, "name": d["name"]})
        return result

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start(self, device_index: int, sample_rate: int, channels: int) -> None:
        # Drain any leftover chunks from a previous session
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break

        self.sample_rate = sample_rate
        self.channels = channels

        self._stream = sd.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        # indata is a view into a PortAudio buffer — always copy before queuing
        self._chunk_queue.put(indata.copy())

    def stop(self) -> float:
        """
        Stop the stream and return time.perf_counter() immediately after stopping.
        The WAV is NOT written here; call save_wav() separately.
        """
        stop_perf = time.perf_counter()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return stop_perf

    def save_wav(self, filepath: str) -> int:
        """
        Drain the chunk queue, concatenate, and write a WAV file.
        Returns the total number of frames saved.
        """
        chunks: list[np.ndarray] = []
        while not self._chunk_queue.empty():
            try:
                chunks.append(self._chunk_queue.get_nowait())
            except queue.Empty:
                break

        if chunks:
            audio_data = np.concatenate(chunks, axis=0)
        else:
            audio_data = np.zeros((0, self.channels), dtype="int16")

        sf.write(filepath, audio_data, self.sample_rate, subtype="PCM_16")
        return len(audio_data)


# ---------------------------------------------------------------------------
# Post-processing: raw MIDI events → CSV rows
# ---------------------------------------------------------------------------

def process_midi_events(
    raw_events: list[tuple[float, list[int]]],
    session_start_perf: float,
    session_start_utc: datetime.datetime,
    session_stop_perf: float,
) -> list[dict]:
    """
    Convert raw (perf_counter, message_bytes) pairs into CSV-ready dicts.

    Duration calculation:
      - Each note_on is matched with its corresponding note_off.
      - A note_on with velocity=0 is treated as note_off (MIDI spec).
      - Notes still held when the session stops get duration = (stop_perf - note_on_perf).

    Returns a list of row dicts sorted by timing_seconds (insertion order is already
    chronological, so no explicit sort is needed).
    """
    # pending_notes[(channel, note)] = (on_perf_counter, row_index_in_rows)
    pending_notes: dict[tuple[int, int], tuple[float, int]] = {}
    rows: list[dict] = []

    for perf_t, message in raw_events:
        if not message:
            continue

        status = message[0]
        d1 = message[1] if len(message) > 1 else 0
        d2 = message[2] if len(message) > 2 else 0

        event_type, channel = parse_midi_status(status)

        # MIDI spec: note_on with velocity=0 is a note_off
        if event_type == "note_on" and d2 == 0:
            event_type = "note_off"

        timing = perf_t - session_start_perf
        iso_ts = make_iso_timestamp(timing, session_start_utc)

        is_note_event = event_type in ("note_on", "note_off", "aftertouch")

        row: dict = {
            "internet_timestamp": iso_ts,
            "event_type": event_type,
            "midi_note": d1 if is_note_event else "",
            "note_name": midi_note_to_name(d1) if is_note_event else "",
            "velocity": d2 if event_type in ("note_on", "note_off") else "",
            "duration_seconds": "",
            "timing_seconds": round(timing, 6),
        }

        if event_type == "note_on":
            key = (channel, d1)
            pending_notes[key] = (perf_t, len(rows))
            rows.append(row)

        elif event_type == "note_off":
            key = (channel, d1)
            if key in pending_notes:
                on_perf, on_idx = pending_notes.pop(key)
                rows[on_idx]["duration_seconds"] = round(perf_t - on_perf, 6)
            rows.append(row)

        else:
            rows.append(row)

    # Notes held at stop: fill in duration up to stop time
    for (_ch, _note), (on_perf, on_idx) in pending_notes.items():
        rows[on_idx]["duration_seconds"] = round(session_stop_perf - on_perf, 6)

    return rows


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

MIDI_CSV_FIELDS = [
    "internet_timestamp",
    "event_type",
    "midi_note",
    "note_name",
    "velocity",
    "duration_seconds",
    "timing_seconds",
]

AUDIO_CSV_FIELDS = [
    "wav_filename",
    "start_timestamp",
    "end_timestamp",
    "sample_rate",
    "channels",
    "duration_seconds",
]


def save_midi_csv(rows: list[dict], filepath: str) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MIDI_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_audio_csv(
    wav_filename: str,
    start_ts: str,
    end_ts: str,
    sample_rate: int,
    channels: int,
    duration_seconds: float,
    filepath: str,
) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIO_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "wav_filename": wav_filename,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_seconds": round(duration_seconds, 6),
            }
        )


# ---------------------------------------------------------------------------
# GUI — RecorderApp
# ---------------------------------------------------------------------------

class RecorderApp(tk.Tk):
    """Main tkinter application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("MIDI + Audio Synchronized Recorder")
        self.resizable(False, False)

        # Recording state
        self._recording = False
        self._midi_rec = MidiRecorder()
        self._audio_rec = AudioRecorder()
        self._session_start_perf: float = 0.0
        self._session_start_utc: datetime.datetime | None = None
        self._session_stop_perf: float = 0.0
        self._ntp_offset: float = 0.0

        # Device lists populated by _refresh_devices()
        self._midi_ports: list[str] = []
        self._audio_devices: list[dict] = []
        self._midi_active: bool = False
        self._audio_active: bool = False

        # Session output metadata (set in _start_recording)
        self._session_out_dir: str = ""
        self._session_tag: str = ""
        self._session_sr: int = 44100
        self._session_ch: int = 1

        # Thread-safe queue for worker → GUI updates
        self._gui_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._refresh_devices()
        self.after(100, self._poll_gui_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ---- MIDI device ----
        midi_frame = ttk.LabelFrame(self, text="MIDI Device")
        midi_frame.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self._midi_var = tk.StringVar()
        self._midi_combo = ttk.Combobox(
            midi_frame, textvariable=self._midi_var, width=44, state="readonly"
        )
        self._midi_combo.grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(midi_frame, text="Refresh", command=self._refresh_devices).grid(
            row=0, column=1, padx=4
        )
        self._midi_ind = tk.Label(midi_frame, text="", font=("Segoe UI", 10))
        self._midi_ind.grid(row=0, column=2, padx=6)

        # ---- Audio device ----
        audio_frame = ttk.LabelFrame(self, text="Audio Input Device")
        audio_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        self._audio_var = tk.StringVar()
        self._audio_combo = ttk.Combobox(
            audio_frame, textvariable=self._audio_var, width=44, state="readonly"
        )
        self._audio_combo.grid(row=0, column=0, padx=4, pady=4)
        self._audio_ind = tk.Label(audio_frame, text="", font=("Segoe UI", 10))
        self._audio_ind.grid(row=0, column=1, padx=6)

        # ---- Audio settings ----
        settings_frame = ttk.LabelFrame(self, text="Audio Settings")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(settings_frame, text="Sample Rate:").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        self._sr_var = tk.StringVar(value="44100")
        for col, sr in enumerate(["44100", "48000"], start=1):
            ttk.Radiobutton(
                settings_frame, text=f"{sr} Hz", variable=self._sr_var, value=sr
            ).grid(row=0, column=col, padx=4)

        ttk.Label(settings_frame, text="Channels:").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )
        self._ch_var = tk.StringVar(value="1")
        ttk.Radiobutton(
            settings_frame, text="Mono", variable=self._ch_var, value="1"
        ).grid(row=1, column=1, padx=4)
        ttk.Radiobutton(
            settings_frame, text="Stereo", variable=self._ch_var, value="2"
        ).grid(row=1, column=2, padx=4)

        # ---- Timestamp mode ----
        ts_frame = ttk.LabelFrame(self, text="Timestamp Mode")
        ts_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        self._ts_mode = tk.StringVar(value="system")
        ttk.Radiobutton(
            ts_frame,
            text="System clock (UTC)",
            variable=self._ts_mode,
            value="system",
            command=self._on_ts_mode_change,
        ).grid(row=0, column=0, padx=4, pady=2)
        ttk.Radiobutton(
            ts_frame,
            text="NTP-corrected",
            variable=self._ts_mode,
            value="ntp",
            command=self._on_ts_mode_change,
        ).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(ts_frame, text="NTP Server:").grid(row=0, column=2, padx=4)
        self._ntp_server_var = tk.StringVar(value="pool.ntp.org")
        self._ntp_entry = ttk.Entry(
            ts_frame, textvariable=self._ntp_server_var, width=22, state="disabled"
        )
        self._ntp_entry.grid(row=0, column=3, padx=4)

        # ---- Output folder ----
        out_frame = ttk.LabelFrame(self, text="Output Folder")
        out_frame.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)

        self._out_dir_var = tk.StringVar(value=os.getcwd())
        ttk.Entry(out_frame, textvariable=self._out_dir_var, width=44).grid(
            row=0, column=0, padx=4, pady=4
        )
        ttk.Button(out_frame, text="Browse…", command=self._browse_dir).grid(
            row=0, column=1, padx=4
        )

        # ---- Start / Stop ----
        self._start_btn = ttk.Button(
            self, text="▶  START", command=self._toggle_recording, width=22
        )
        self._start_btn.grid(row=5, column=0, columnspan=2, pady=8)

        # ---- Status label ----
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            self, textvariable=self._status_var, font=("Segoe UI", 9, "italic")
        ).grid(row=6, column=0, columnspan=2, pady=(0, 4))

        # ---- Log ----
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", **pad)

        self._log = scrolledtext.ScrolledText(
            log_frame,
            width=64,
            height=10,
            state="disabled",
            font=("Consolas", 8),
        )
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

        self.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_ts_mode_change(self) -> None:
        state = "normal" if self._ts_mode.get() == "ntp" else "disabled"
        self._ntp_entry.configure(state=state)

    def _refresh_devices(self) -> None:
        # MIDI ports
        self._midi_ports = MidiRecorder.get_port_names()
        self._midi_active = bool(self._midi_ports)
        self._midi_combo["values"] = (
            self._midi_ports if self._midi_ports else ["(no MIDI devices found)"]
        )
        if self._midi_ports:
            self._midi_combo.current(0)
            self._midi_combo.configure(state="readonly")
        else:
            self._midi_combo.configure(state="disabled")
        if self._midi_active:
            self._midi_ind.configure(text=f"✔ {len(self._midi_ports)} device(s)", fg="#2e7d32")
        else:
            self._midi_ind.configure(text="— not found", fg="#9e9e9e")

        # Audio input devices
        self._audio_devices = AudioRecorder.get_device_list()
        self._audio_active = bool(self._audio_devices)
        audio_labels = [f"[{d['index']}] {d['name']}" for d in self._audio_devices]
        self._audio_combo["values"] = (
            audio_labels if audio_labels else ["(no input devices found)"]
        )
        if self._audio_devices:
            self._audio_combo.current(0)
            self._audio_combo.configure(state="readonly")
        else:
            self._audio_combo.configure(state="disabled")
        if self._audio_active:
            self._audio_ind.configure(text=f"✔ {len(self._audio_devices)} device(s)", fg="#2e7d32")
        else:
            self._audio_ind.configure(text="— not found", fg="#9e9e9e")

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=self._out_dir_var.get())
        if d:
            self._out_dir_var.set(d)

    def _log_msg(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    # ------------------------------------------------------------------
    # GUI queue polling  (worker thread → main thread)
    # ------------------------------------------------------------------

    def _poll_gui_queue(self) -> None:
        try:
            while True:
                item = self._gui_queue.get_nowait()
                action = item[0]
                if action == "log":
                    self._log_msg(item[1])
                elif action == "status":
                    self._status_var.set(item[1])
                elif action == "stop_done":
                    self._on_stop_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_gui_queue)

    # ------------------------------------------------------------------
    # Recording toggle
    # ------------------------------------------------------------------

    def _toggle_recording(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        use_midi = self._midi_active
        use_audio = self._audio_active

        if not use_midi and not use_audio:
            messagebox.showerror("Error", "No MIDI or audio device found. Connect at least one device and click Refresh.")
            return

        # --- MIDI device ---
        midi_idx = self._midi_combo.current() if use_midi else -1

        # --- Audio device ---
        audio_list_idx = self._audio_combo.current() if use_audio else -1
        audio_sd_idx = self._audio_devices[audio_list_idx]["index"] if use_audio and audio_list_idx >= 0 else -1

        sample_rate = int(self._sr_var.get())
        channels = int(self._ch_var.get())

        # --- Validate output folder ---
        out_dir = self._out_dir_var.get().strip()
        if not os.path.isdir(out_dir):
            messagebox.showerror("Error", f"Output folder does not exist:\n{out_dir}")
            return

        # --- Timestamp setup ---
        use_ntp = self._ts_mode.get() == "ntp"
        ntp_offset = 0.0
        if use_ntp:
            ntp_server = self._ntp_server_var.get().strip()
            self._log_msg(f"Querying NTP server: {ntp_server} …")
            try:
                ntp_offset = get_ntp_offset(ntp_server)
                self._log_msg(f"NTP offset: {ntp_offset:+.4f} s")
            except Exception as exc:
                proceed = messagebox.askyesno(
                    "NTP Error",
                    f"NTP query failed:\n{exc}\n\nContinue with system clock?",
                )
                if not proceed:
                    return
                use_ntp = False
                ntp_offset = 0.0

        self._ntp_offset = ntp_offset

        # --- Session anchor: record perf_counter and UTC together ---
        # Taking perf_counter and utcnow() as close together as possible.
        self._session_start_perf = time.perf_counter()
        raw_utc = datetime.datetime.utcnow()
        self._session_start_utc = raw_utc + datetime.timedelta(seconds=ntp_offset)

        # --- Save session metadata ---
        self._session_out_dir = out_dir
        self._session_tag = raw_utc.strftime("%Y%m%d_%H%M%S")
        self._session_sr = sample_rate
        self._session_ch = channels

        # --- Open MIDI port ---
        if use_midi:
            try:
                self._midi_rec.start(midi_idx, self._session_start_perf)
            except Exception as exc:
                messagebox.showerror("MIDI Error", f"Cannot open MIDI port:\n{exc}")
                return

        # --- Open audio stream ---
        if use_audio:
            try:
                self._audio_rec.start(audio_sd_idx, sample_rate, channels)
            except Exception as exc:
                if use_midi:
                    self._midi_rec.stop()
                messagebox.showerror("Audio Error", f"Cannot open audio device:\n{exc}")
                return

        self._recording = True
        self._start_btn.configure(text="⏹  STOP")

        start_ts = self._session_start_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        self._status_var.set(f"Recording…  started {start_ts}")
        self._log_msg(f"[START] {start_ts}")
        if use_midi:
            self._log_msg(f"  MIDI  → port {midi_idx}: {self._midi_ports[midi_idx]}")
        else:
            self._log_msg("  MIDI  → not connected (skipped)")
        if use_audio:
            self._log_msg(
                f"  Audio → [{audio_sd_idx}] "
                f"{self._audio_devices[audio_list_idx]['name']}  "
                f"{sample_rate} Hz  {'Mono' if channels == 1 else 'Stereo'}"
            )
        else:
            self._log_msg("  Audio → not connected (skipped)")
        ts_mode_label = f"NTP ({self._ntp_server_var.get()})" if use_ntp else "System clock (UTC)"
        self._log_msg(f"  Clock  → {ts_mode_label}")

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        self._start_btn.configure(state="disabled", text="Processing…")
        self._status_var.set("Stopping — saving files…")
        # Off-load post-processing to a worker thread so the GUI stays responsive
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        """Worker thread: stop recorders, post-process, save files, update GUI."""
        try:
            # Stop active streams; use fallback perf_counter if audio is not active
            if self._audio_active:
                stop_perf = self._audio_rec.stop()
            else:
                stop_perf = time.perf_counter()
            if self._midi_active:
                self._midi_rec.stop()
            self._session_stop_perf = stop_perf

            tag = self._session_tag
            out_dir = self._session_out_dir

            # ---- Build session timestamps ----
            start_ts = self._session_start_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
            duration_s = stop_perf - self._session_start_perf
            stop_utc = self._session_start_utc + datetime.timedelta(seconds=duration_s)
            stop_ts = stop_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

            self._gui_queue.put(("log", f"[STOP]  {stop_ts}"))
            self._gui_queue.put(("log", f"  Duration     : {duration_s:.3f} s"))

            # ---- Save WAV + audio CSV (only if audio was active) ----
            if self._audio_active:
                wav_name = f"{tag}_recording.wav"
                wav_path = os.path.join(out_dir, wav_name)
                frames_saved = self._audio_rec.save_wav(wav_path)
                audio_csv_path = os.path.join(out_dir, f"{tag}_audio.csv")
                save_audio_csv(
                    wav_filename=wav_name,
                    start_ts=start_ts,
                    end_ts=stop_ts,
                    sample_rate=self._session_sr,
                    channels=self._session_ch,
                    duration_seconds=duration_s,
                    filepath=audio_csv_path,
                )
                self._gui_queue.put(("log", f"  Audio frames : {frames_saved}"))
                self._gui_queue.put(("log", f"  WAV          : {wav_path}"))
                self._gui_queue.put(("log", f"  Audio CSV    : {audio_csv_path}"))

            # ---- Post-process MIDI + save CSV (only if MIDI was active) ----
            if self._midi_active:
                raw_events = self._midi_rec.get_events()
                midi_rows = process_midi_events(
                    raw_events=raw_events,
                    session_start_perf=self._session_start_perf,
                    session_start_utc=self._session_start_utc,
                    session_stop_perf=self._session_stop_perf,
                )
                midi_csv_path = os.path.join(out_dir, f"{tag}_midi.csv")
                save_midi_csv(midi_rows, midi_csv_path)
                self._gui_queue.put(
                    ("log", f"  MIDI CSV     : {midi_csv_path}  ({len(midi_rows)} rows)")
                )
                midi_summary = f"{len(midi_rows)} MIDI events"
            else:
                midi_summary = "no MIDI"
            audio_summary = f"{duration_s:.2f} s audio" if self._audio_active else "no audio"
            self._gui_queue.put(("status", f"Done — {midi_summary} · {audio_summary}"))
            self._gui_queue.put(("stop_done",))

        except Exception as exc:
            self._gui_queue.put(("log", f"[ERROR] {exc}\n{traceback.format_exc()}"))
            self._gui_queue.put(("status", f"Error: {exc}"))
            self._gui_queue.put(("stop_done",))

    def _on_stop_done(self) -> None:
        """Called from main thread via queue poll after the worker finishes."""
        self._recording = False
        self._start_btn.configure(state="normal", text="▶  START")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = RecorderApp()
    app.mainloop()
