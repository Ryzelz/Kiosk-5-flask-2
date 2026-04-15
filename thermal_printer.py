"""
Thermal Printer Driver for QR204 on Raspberry Pi 5.

Communicates over USB serial using ESC/POS commands.
Default device: /dev/ttyUSB0 (typical for USB-to-serial adapters).
Adjust SERIAL_PORT or pass it as an argument if your wiring differs.

Usage:
    python thermal_printer.py
    python thermal_printer.py --port /dev/ttyUSB0 --baud 9600
"""

import argparse
import time

try:
    import serial
except ImportError:
    serial = None

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
DEFAULT_PORT = '/dev/ttyUSB0'
DEFAULT_BAUD = 9600
DEFAULT_TIMEOUT = 5


class QR204Printer:
    """Driver for the QR204 thermal printer over serial."""

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUD, timeout=DEFAULT_TIMEOUT):
        if serial is None:
            raise RuntimeError(
                'pyserial is required. Install it with: pip install pyserial'
            )

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._connection = None

    # ── Connection management ────────────────────────────────────────────

    def open(self):
        if self._connection and self._connection.is_open:
            return

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
    parser.add_argument('--port', default=DEFAULT_PORT, help=f'Serial port (default: {DEFAULT_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD, help=f'Baud rate (default: {DEFAULT_BAUD})')
    parser.add_argument('--text', type=str, default=None, help='Print a single line of text and exit')
    args = parser.parse_args()

    with QR204Printer(port=args.port, baudrate=args.baud) as printer:
        if args.text:
            printer.println(args.text)
            printer.feed(3)
            print(f'Printed: {args.text}')
        else:
            demo_receipt(printer)
            print('Demo receipt printed successfully.')


if __name__ == '__main__':
    main()
