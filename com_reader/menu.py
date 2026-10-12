"""Optional arrow-key menu. No serial connection is opened just by launching it."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Protocol

from .cli import DEFAULT_CONFIG
from .config import Config, ConfigError
from .device import Device, DeviceError, Sample, SampleTimeout
from .protocol import configure_command
from .simulation import SimulatedSerial


class MenuUI(Protocol):
    def select(self, title: str, choices: list[tuple[str, str | None]]) -> str | None: ...
    def text(self, title: str, default: str = "") -> str | None: ...
    def confirm(self, title: str) -> bool: ...
    def report(self, message: str) -> None: ...


class MenuDevice(Protocol):
    @property
    def is_running(self) -> bool: ...
    def start(self) -> None: ...
    def close(self) -> None: ...
    def wait_next_sample(self) -> Sample: ...


class QuestionaryUI:
    def __init__(self):
        try:
            import questionary
            from prompt_toolkit.application import create_app_session
            from prompt_toolkit.input.defaults import create_input
            from prompt_toolkit.output.defaults import create_output
            from prompt_toolkit.patch_stdout import StdoutProxy
        except ImportError as exc:
            raise RuntimeError("Установите меню: python -m pip install -r requirements-menu.txt") from exc
        self.questionary = questionary
        self._contexts = ExitStack()
        try:
            self._input = create_input(stdin=sys.stdin)
            self._contexts.callback(self._input.close)
            self._output = create_output(stdout=sys.stderr)
            self._contexts.enter_context(create_app_session(input=self._input, output=self._output))
            # Capture the same app session as the prompts: worker logs are rendered
            # above the active menu, on stderr, rather than corrupting its selection.
            self._proxy = self._contexts.enter_context(StdoutProxy(raw=True))
        except BaseException:
            self._contexts.close()
            raise

    def select(self, title, choices):
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

        back = object()  # Questionary replaces value=None with the displayed title.
        question = self.questionary.select(
            title, choices=[self.questionary.Choice(
                label, value=value if value is not None else back,
                shortcut_key="q" if value is None else True,
            ) for label, value in choices],
            use_shortcuts=True,
            instruction="↑/↓ — выбор, Enter — выполнить, q — назад/выход, Ctrl+C — выход",
            input=self._input, output=self._output,
        )
        shortcuts = KeyBindings()

        @shortcuts.add("q", eager=True)
        def leave(event):
            event.app.exit(result=back)

        question.application.key_bindings = merge_key_bindings([
            question.application.key_bindings, shortcuts,
        ])
        selected = question.unsafe_ask()
        return None if selected is back else selected

    def text(self, title, default=""):
        return self.questionary.text(title, default=default, input=self._input, output=self._output).unsafe_ask()

    def confirm(self, title):
        return self.questionary.confirm(title, default=False, input=self._input, output=self._output).unsafe_ask()

    def report(self, message):
        self._proxy.write(message + "\n")
        self._proxy.flush()

    def close(self):
        self._contexts.close()


def available_ports():
    from serial.tools import list_ports
    return sorted(list_ports.comports(), key=lambda port: port.device)


class MenuApplication:
    def __init__(self, config: Config, ui: MenuUI, *, config_path: Path = DEFAULT_CONFIG,
                 trace_path: Path | None = None, output=None,
                 device_factory: Callable[..., MenuDevice] = Device, ports_provider=available_ports):
        self.config = config
        self.config_path = config_path
        self.trace_path = trace_path
        self.ui = ui
        self.output = sys.stdout if output is None else output
        self.device_factory = device_factory
        self.ports_provider = ports_provider
        self.device: MenuDevice | None = None
        self.demo = False
        self.exit_code = 0

    def _disconnected(self) -> bool:
        if self.device is not None:
            self.ui.report("Сначала отключите текущую сессию. Настройки работающего прибора не меняются.")
            return False
        return True

    def show_settings(self):
        m, s = self.config.measurement, self.config.serial
        self.ui.report(f"JSON: {self.config_path}\nПорт: {s.port}; {s.baudrate} бод, 8N1. "
                       f"Режим: Single.\nФизические каналы / порядок CSV: {', '.join(map(str, m.channels))}.\n"
                       f"Интервал: {m.interval_seconds} с; фильтр: {m.filter_hz} Гц.\n"
                       f"Трассировка: {self.trace_path or 'выключена'}.\n"
                       "Единицы неизвестны; signed int16 без пересчёта и поправок нуля.")
        if self.config.legacy.get("original_mode") == "cross":
            self.ui.report("Исходный профиль — Cross; текущий Single не считается физически эквивалентным.")
        self.ui.report(f"RTS={s.rts}, DTR={s.dtr}; null означает defaults pyserial. "
                       "Аппаратная совместимость не подтверждена.")

    def list_ports(self):
        ports = self.ports_provider()
        if not ports:
            self.ui.report("COM-порты не найдены. Проверьте USB-подключение и драйвер прибора.")
        for port in ports:
            self.ui.report(f"{port.device}: {port.description}\n  HWID: {port.hwid}")
        return ports

    def choose_port(self):
        if not self._disconnected():
            return
        ports = self.list_ports()
        choices = [(f"{p.device} — {p.description}", p.device) for p in ports]
        choices.extend([("Ввести порт вручную", "manual"), ("Назад", None)])
        selected = self.ui.select("Выберите порт", choices)
        if selected == "manual":
            selected = self.ui.text("Порт, например COM4", self.config.serial.port)
        if selected is None:
            return
        selected = selected.strip()
        if not selected or any(ord(c) < 32 for c in selected):
            self.ui.report("Имя порта должно быть непустым и без управляющих символов.")
            return
        self.config = replace(self.config, serial=replace(self.config.serial, port=selected))
        self.ui.report(f"Выбран {selected}. Изменение действует только в текущем меню; JSON не перезаписан.")

    def connect(self, demo=False):
        if not self._disconnected():
            return
        self.show_settings()
        if not demo and not self.ui.confirm("Отправить STOP, настройки Single и START? Закройте старую программу."):
            self.ui.report("Подключение отменено; порт не открывался.")
            return
        if demo:
            self.ui.report("ДЕМО: только синтетические значения, реальный COM-порт не открывается.")
        factory = None
        if demo:
            factory = lambda s, read_timeout, write_timeout: SimulatedSerial(
                read_timeout, self.config.protocol.data_terminators[0]
            )
        self.device = self.device_factory(self.config, log=self.ui.report, trace_path=self.trace_path,
                                          serial_factory=factory)
        self.demo = demo
        try:
            self.device.start()
        except BaseException:
            self.disconnect()
            raise
        self.ui.report("Сессия запущена. Для показаний откройте «Измерения → Снять один набор».")

    def disconnect(self):
        if self.device is None:
            self.ui.report("Активной сессии нет.")
            return
        try:
            self.device.close()
        except DeviceError as exc:
            self.exit_code = 1
            self.ui.report(f"Ошибка завершения: {exc}")
            if self.device.is_running:
                self.ui.report("Поток ещё работает. Новое подключение заблокировано; повторите отключение.")
                return
        self.device = None
        self.demo = False
        self.ui.report("Сессия закрыта.")

    def acquire(self):
        if self.device is None:
            self.ui.report("Сначала выберите «Подключение → Подключиться» или «Деморежим».")
            return
        self.ui.report(f"Ожидаю следующий полный принятый набор, до {self.config.protocol.sample_timeout_seconds:g} с. "
                       "Это программный запрос, не аппаратный триггер.")
        try:
            sample = self.device.wait_next_sample()
        except SampleTimeout as exc:
            self.ui.report(str(exc))
            return
        except DeviceError:
            self.disconnect()
            raise
        writer = csv.writer(self.output, lineterminator="\n")
        writer.writerow([sample.timestamp.isoformat(timespec="milliseconds")] + list(sample.values))
        self.output.flush()
        self.ui.report(f"Набор №{sample.sequence}: {len(sample.values)} значений выведено в CSV stdout.")

    def load_config(self):
        if not self._disconnected():
            return
        name = self.ui.text("Путь к JSON", str(self.config_path))
        if not name or not name.strip():
            return
        path = Path(name.strip()).expanduser()
        config = Config.load(path)
        self.config, self.config_path = config, path
        self.ui.report("JSON загружен. Исходные файлы не изменены.")
        self.show_settings()

    def set_trace(self):
        if not self._disconnected():
            return
        name = self.ui.text("Файл трассировки .jsonl (пустая строка — выключить)",
                            str(self.trace_path) if self.trace_path else "wire_menu.jsonl")
        if name is None:
            return
        path = Path(name.strip()).expanduser() if name.strip() else None
        if path is not None and (path.suffix.lower() != ".jsonl" or not path.parent.is_dir()):
            self.ui.report("Нужен файл с расширением .jsonl в существующей директории.")
            return
        self.trace_path = path
        self.ui.report(f"Трассировка для следующей сессии: {path or 'выключена'}. Существующий файл дополняется.")

    def submenu(self, title, actions):
        choices = [(label, key) for key, label, action in actions] + [("Назад", None)]
        while True:
            selected = self.ui.select(title, choices)
            if selected is None:
                return
            action = next(action for key, label, action in actions if key == selected)
            try:
                action()
            except (DeviceError, OSError, ConfigError, ImportError) as exc:
                self.ui.report(f"Ошибка: {exc}")
                if isinstance(exc, DeviceError):
                    self.exit_code = 1

    def run(self) -> int:
        self.ui.report("Меню не открывает порт автоматически. CSV — stdout, меню и диагностика — stderr.")
        try:
            while True:
                state = "ДЕМО" if self.device is not None and self.demo else (
                    self.config.serial.port if self.device is not None else "не подключено")
                selected = self.ui.select(f"EchM · Single · {state}", [
                    ("Подключение", "connection"), ("Настройки", "settings"),
                    ("Измерения", "measurements"), ("Диагностика", "diagnostics"), ("Выход", None),
                ])
                if selected is None:
                    break
                if selected == "connection":
                    self.submenu("Подключение", [
                        ("ports", "Показать COM / USB–COM устройства", self.list_ports),
                        ("choose", "Выбрать порт", self.choose_port),
                        ("connect", "Подключиться к прибору", self.connect),
                        ("demo", "Деморежим (без прибора)", lambda: self.connect(demo=True)),
                        ("disconnect", "Отключиться", self.disconnect),
                    ])
                elif selected == "settings":
                    self.submenu("Настройки", [
                        ("show", "Показать текущие настройки", self.show_settings),
                        ("load", "Загрузить другой JSON", self.load_config),
                    ])
                elif selected == "measurements":
                    self.submenu("Измерения", [("sample", "Снять один набор", self.acquire)])
                elif selected == "diagnostics":
                    self.submenu("Диагностика", [
                        ("packet", "Проверить настройки / показать команду", lambda: self.ui.report(
                            f"Команда настройки ({len(configure_command(self.config.measurement))} байт): "
                            f"{configure_command(self.config.measurement)!r}")),
                        ("trace", "Настроить запись RX/TX", self.set_trace),
                    ])
        except KeyboardInterrupt:
            self.exit_code = 130
            self.ui.report("Ctrl+C: завершаю сессию.")
        except EOFError:
            self.ui.report("Ввод закрыт; завершаю сессию.")
        finally:
            if self.device is not None:
                self.disconnect()
        return self.exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Отдельное интерактивное меню EchM; порт открывается только по выбору пользователя.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", help="Переопределить начальный порт без изменения JSON")
    parser.add_argument("--trace", type=Path, help="Локальный JSONL для следующей сессии")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        if args.port is not None:
            if not args.port.strip():
                raise ConfigError("--port не должен быть пустым")
            config = replace(config, serial=replace(config.serial, port=args.port.strip()))
        if args.trace and (args.trace.suffix.lower() != ".jsonl" or not args.trace.parent.is_dir()):
            raise ConfigError("--trace: нужен файл .jsonl в существующей директории")
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            raise RuntimeError("Меню требует интерактивный терминал. Для pipe используйте python -m com_reader.")
        ui = QuestionaryUI()
        try:
            return MenuApplication(config, ui, config_path=args.config, trace_path=args.trace).run()
        finally:
            ui.close()
    except (ConfigError, OSError, RuntimeError) as exc:
        print(f"Ошибка меню: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
