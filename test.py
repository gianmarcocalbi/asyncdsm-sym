from curses import wrapper
import time

def main(stdscr):
    # Clear screen
    stdscr.clear()

    # This raises ZeroDivisionError when i == 10.
    for i in range(0, 10):
        stdscr.addstr(0, 0, 'Time {}'.format(i))
        stdscr.refresh()
        time.sleep(1)

    stdscr.getkey()

wrapper(main)