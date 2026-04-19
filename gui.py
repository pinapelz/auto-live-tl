import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from typing import Iterable, List, Tuple, Dict, Any
import sounddevice as sd


def select_settings(
    settings: Dict[str, Any],
    input_devices: List[Tuple[int, Dict[str, Any]]],
    default_settings: Dict[str, Any],
    model_choices: Iterable[str],
    device_choices: Iterable[str],
    compute_choices: Iterable[str],
    task_choices: Iterable[str],
) -> Dict[str, Any]:
    if not input_devices:
        raise RuntimeError("No audio input devices found.")

    def get_value(key: str, fallback: Any) -> Any:
        return settings.get(key, default_settings.get(key, fallback))

    root = tk.Tk()
    root.title("Audio & Whisper Settings")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def add_row(row: int, label_text: str, widget: tk.Widget) -> None:
        label = ttk.Label(frame, text=label_text)
        label.grid(row=row, column=0, sticky="w", pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    device_options = [
        f"[{idx}] {dev['name']} ({dev.get('max_input_channels', 0)} ch)"
        for idx, dev in input_devices
    ]
    device_names = [dev["name"] for _idx, dev in input_devices]
    device_combo = ttk.Combobox(frame, values=device_options, state="readonly", width=60)

    default_device_name = get_value("audio_device_name", "")
    if default_device_name in device_names:
        device_combo.current(device_names.index(default_device_name))
    else:
        device_combo.current(0)
    add_row(0, "Audio input device:", device_combo)

    model_var = tk.StringVar(value=get_value("model_name", "medium"))
    model_combo = ttk.Combobox(frame, values=list(model_choices), textvariable=model_var)
    model_combo.set(model_var.get())
    add_row(1, "Model:", model_combo)

    device_type_var = tk.StringVar(value=get_value("device", "cpu"))
    device_type_combo = ttk.Combobox(
        frame, values=list(device_choices), textvariable=device_type_var, state="readonly"
    )
    device_type_combo.set(device_type_var.get())
    add_row(2, "Compute device:", device_type_combo)

    compute_type_var = tk.StringVar(value=get_value("compute_type", "int8"))
    compute_type_combo = ttk.Combobox(
        frame, values=list(compute_choices), textvariable=compute_type_var
    )
    compute_type_combo.set(compute_type_var.get())
    add_row(3, "Compute type:", compute_type_combo)

    task_var = tk.StringVar(value=get_value("task", "translate"))
    task_combo = ttk.Combobox(frame, values=list(task_choices), textvariable=task_var, state="readonly")
    task_combo.set(task_var.get())
    add_row(4, "Task:", task_combo)

    beam_size_var = tk.StringVar(value=str(get_value("beam_size", 3)))
    beam_entry = ttk.Entry(frame, textvariable=beam_size_var, width=10)
    add_row(5, "Beam size:", beam_entry)

    language_var = tk.StringVar(value=get_value("language", ""))
    language_entry = ttk.Entry(frame, textvariable=language_var)
    add_row(6, "Language (optional):", language_entry)

    context_seconds_var = tk.StringVar(value=str(get_value("context_seconds", 10)))
    context_entry = ttk.Entry(frame, textvariable=context_seconds_var, width=10)
    add_row(7, "Context seconds:", context_entry)

    update_interval_var = tk.StringVar(value=str(get_value("update_interval_seconds", 2)))
    update_interval_entry = ttk.Entry(frame, textvariable=update_interval_var, width=10)
    add_row(8, "Update interval (s):", update_interval_entry)

    button_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    button_frame.pack(fill="x")

    selected_settings: Dict[str, Any] = {}

    def on_ok() -> None:
        nonlocal selected_settings
        selection = device_combo.current()
        if selection < 0:
            messagebox.showwarning("Select a device", "Please select an audio input device.")
            return
        model_name = model_var.get().strip()
        if not model_name:
            messagebox.showwarning("Model required", "Please select or enter a model name.")
            return
        try:
            beam_size = int(beam_size_var.get().strip())
            if beam_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid beam size", "Beam size must be a positive integer.")
            return
        try:
            context_seconds = float(context_seconds_var.get().strip())
            if context_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid context seconds", "Context seconds must be a positive number.")
            return
        try:
            update_interval_seconds = float(update_interval_var.get().strip())
            if update_interval_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Invalid update interval", "Update interval must be a positive number."
            )
            return
        selected_settings = {
            "audio_device_name": device_names[selection],
            "model_name": model_name,
            "device": device_type_var.get().strip() or "cpu",
            "compute_type": compute_type_var.get().strip() or "int8",
            "task": task_var.get().strip() or "translate",
            "beam_size": beam_size,
            "language": language_var.get().strip(),
            "context_seconds": context_seconds,
            "update_interval_seconds": update_interval_seconds,
        }
        root.quit()

    def on_cancel() -> None:
        root.quit()

    ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
    cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    ok_button.pack(side="left", padx=(0, 6))
    cancel_button.pack(side="left")

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    root.destroy()

    if not selected_settings:
        raise SystemExit("No settings selected.")
    return selected_settings


def prompt_input_sample_rate(device_index: int, common_rates: Iterable[int] | None = None) -> int:
    rates = list(common_rates) if common_rates is not None else [48000, 44100, 32000, 24000, 22050, 16000, 12000, 8000]
    root = tk.Tk()
    root.withdraw()
    try:
        while True:
            prompt = (
                "Enter an input sample rate in Hz.\n"
                f"Common values: {', '.join(str(r) for r in rates)}"
            )
            raw = simpledialog.askstring("Select Sample Rate", prompt, parent=root)
            if raw is None:
                raise sd.PortAudioError("No supported input sample rate found for selected device.")
            raw = raw.strip()
            if not raw:
                continue
            try:
                rate = int(float(raw))
            except ValueError:
                messagebox.showwarning("Invalid value", "Sample rate must be a number.", parent=root)
                continue
            try:
                sd.check_input_settings(device=device_index, channels=1, samplerate=rate, dtype="float32")
                return rate
            except sd.PortAudioError:
                messagebox.showwarning(
                    "Unsupported sample rate",
                    f"{rate} Hz is not supported by the selected device.",
                    parent=root,
                )
    finally:
        root.destroy()