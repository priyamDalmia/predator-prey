import time
import curses


stdscr = curses.initscr()

curses.noecho()
curses.cbreak()
stdscr.addstr(5, 5, "Hello")
stdscr.refresh()
time.sleep(3)


curses.echo()
curses.nocbreak()
stdscr.keypad(False)

curses.endwin()
