"""
Thermal Printer Driver for QR204 on Raspberry Pi 5.

Communicates over USB serial using ESC/POS commands.
The port is auto-detected at startup. You can also pass --port explicitly.

Usage:
    python thermal_printer.py
    python thermal_printer.py --port /dev/ttyUSB0 --baud 9600
    python thermal_printer.py --list-ports
"""

import argparse
import glob
import os
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

# ── ESC/POS command constants ────────────────────────────────────────────────
ESC = b'\x1b'
GS = b'\x1d'

INIT = ESC + b'\x40'                 # Initialize printer
LINE_FEED = b'\x0a'
CUT_PAPER = GS + b'\x56\x00'        # Full cut (if cutter installed)

ALIGN_LEFT = ESC + b'\x61\x00'
ALIGN_CENTER = ESC + b'\x61\x01'
ALIGN_RIGHT = ESC + b'\x61\x02'

BOLD_ON = ESC + b'\x45\x01'
BOLD_OFF = ESC + b'\x45\x00'

DOUBLE_HEIGHT_ON = GS + b'\x21\x10'
DOUBLE_WIDTH_ON = GS + b'\x21\x20'
DOUBLE_SIZE_ON = GS + b'\x21\x30'
NORMAL_SIZE = GS + b'\x21\x00'

UNDERLINE_ON = ESC + b'\x2d\x01'
UNDERLINE_OFF = ESC + b'\x2d\x00'

# Default serial settings for QR204
DEFAULT_PORT = None   # None = auto-detect
DEFAULT_BAUD = 9600
DEFAULT_TIMEOUT = 5

# Candidate port patterns to scan when auto-detecting (Linux + macOS + Windows)
_PORT_CANDIDATES = [
    '/dev/ttyUSB*',
    '/dev/ttyACM*',
    '/dev/usb/lp*',
    '/dev/ttyS*',
    '/dev/tty.usbserial*',
    '/dev/tty.usbmodem*',
]


def find_printer_port():
    """
    Return the first likely serial port for the thermal printer.

    Priority:
    1. ttyUSB* (USB-to-serial adapter, most common for QR204)
    2. ttyACM* (CDC-ACM USB class)
    3. usb/lp* (USB printer class — raw, no serial framing)
    4. Any other port reported by pyserial list_ports
    """
    # Glob-based scan (Linux / macOS)
    for pattern in _PORT_CANDIDATES:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]

    # Fallback: pyserial's own port enumerator (works on all platforms)
    if list_ports:
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        if ports:
            return ports[0].device

    return None


def _list_all_ports():
    """Return a list of all available serial port device strings."""
    found = []
    for pattern in _PORT_CANDIDATES:
        found.extend(sorted(glob.glob(pattern)))
    if list_ports:
        for p in sorted(list_ports.comports(), key=lambda x: x.device):
            if p.device not in found:
                found.append(p.device)
    return found


class QR204Printer:
    """Driver for the QR204 thermal printer over serial."""

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUD, timeout=DEFAULT_TIMEOUT):
        if serial is None:
            raise RuntimeError(
                'pyserial is required. Install it with: pip install pyserial'
            )

        # If no port given, auto-detect
        self.port = port or find_printer_port()
        self.baudrate = baudrate
        self.timeout = timeout
        self._connection = None

    # ── Connection management ────────────────────────────────────────────

    def open(self):
        if self._connection and self._connection.is_open:
            return

        if not self.port:
            available = _list_all_ports()
            hint = '\n  '.join(available) if available else '  (none found — is the printer plugged in?)'
            raise RuntimeError(
                f'No serial port found for the thermal printer.\n'
                f'Available ports:\n  {hint}\n'
                f'Pass --port <device> to specify one explicitly.'
            )

        self._connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        time.sleep(0.5)
        self._write(INIT)

    def close(self):
        if self._connection and self._connection.is_open:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ── Low-level helpers ────────────────────────────────────────────────

    def _write(self, data):
        if not self._connection or not self._connection.is_open:
            raise RuntimeError('Printer connection is not open.')

        if isinstance(data, str):
            data = data.encode('utf-8')

        self._connection.write(data)
        self._connection.flush()

    # ── Public API ───────────────────────────────────────────────────────

    def println(self, text=''):
        """Print a line of text followed by a line feed."""
        self._write(text)
        self._write(LINE_FEED)

    def feed(self, lines=1):
        """Feed blank lines."""
        for _ in range(lines):
            self._write(LINE_FEED)

    def bold(self, enabled=True):
        self._write(BOLD_ON if enabled else BOLD_OFF)

    def underline(self, enabled=True):
        self._write(UNDERLINE_ON if enabled else UNDERLINE_OFF)

    def align(self, position='left'):
        commands = {'left': ALIGN_LEFT, 'center': ALIGN_CENTER, 'right': ALIGN_RIGHT}
        self._write(commands.get(position, ALIGN_LEFT))

    def double_size(self, enabled=True):
        self._write(DOUBLE_SIZE_ON if enabled else NORMAL_SIZE)

    def cut(self):
        self.feed(3)
        self._write(CUT_PAPER)

    def print_separator(self, char='-', width=32):
        self.println(char * width)


def demo_receipt(printer):
    """Print a sample receipt to verify the QR204 is working."""
    printer.align('center')
    printer.bold()
    printer.double_size()
    printer.println('WIDEYE KIOSK')
    printer.double_size(False)
    printer.bold(False)
    printer.println('Your friendly self-service kiosk')
    printer.print_separator()

    printer.align('left')
    printer.println('Item            Qty   Price')
    printer.print_separator()
    printer.println('Cafe Latte        1   P150.00')
    printer.println('Mocha Frappe      2   P340.00')
    printer.print_separator()

    printer.align('right')
    printer.bold()
    printer.println('TOTAL:   P490.00')
    printer.bold(False)

    printer.align('center')
    printer.feed(1)
    printer.println('Thank you for your purchase!')
    printer.println('Please come again.')

    printer.cut()


def main():
    parser = argparse.ArgumentParser(description='QR204 thermal printer driver for Raspberry Pi 5')
    parser.add_argument('--port', default=None, help='Serial port (default: auto-detect)')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD, help=f'Baud rate (default: {DEFAULT_BAUD})')
    parser.add_argument('--text', type=str, default=None, help='Print a single line of text and exit')
    parser.add_argument('--list-ports', action='store_true', help='List all available serial ports and exit')
    args = parser.parse_args()

    if args.list_ports:
        ports = _list_all_ports()
        if ports:
            print('Available serial ports:')
            for p in ports:
                print(f'  {p}')
        else:
            print('No serial ports found. Is the printer plugged in?')
        return

    port = args.port or find_printer_port()
    if port:
        print(f'Using port: {port}')
    else:
        print('ERROR: No serial port detected.')
        print('Plug in the printer and try again, or pass --port <device> explicitly.')
        print('Run with --list-ports to see all available ports.')
        return

    with QR204Printer(port=port, baudrate=args.baud) as printer:
        if args.text:
            printer.println(args.text)
            printer.feed(3)
            print(f'Printed: {args.text}')
        else:
            demo_receipt(printer)
            print('Demo receipt printed successfully.')


if __name__ == '__main__':
    main()
