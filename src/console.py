import curses

class CursesStdout:
    def __init__(self):
        self.screen = curses.initscr()

    def open(self):
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)


    def close(self):
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()


stdout = CursesStdout()


def print(str):
    stdout.screen.addstr("\n" + str)
    stdout.screen.refresh()
