from datetime import datetime
import os
from telegram_send_message import telegram_send as t
import inspect

absolute_path = os.path.dirname(__file__)
config = os.path.join(absolute_path, f'config{os.sep}')
log = os.path.join(absolute_path, f'log{os.sep}')

class LogManager:
    def __init__(self, username):
        self.persistent_log = log + "log_" + username + ".txt"
        # self.temp = log + "temp_" + username + ".txt"
        #
        # if os.path.exists(self.temp):
        #     os.remove(self.temp)
        # # create file
        # open(self.temp, "x")

        self.username = username
        # List of files to skip (without .py extension)
        self.SKIP_FILES = {"integrity_constraint_manager", "directory_manager", "decorator_definition", "contextlib"}

        # if persistent log does not exist, create it
        if not os.path.exists(self.persistent_log):
            open(self.persistent_log, "x")

    def printl(self, s, verbose=0):
        # stack = inspect.stack()
        # # Determine the correct caller:
        # # - If called from a decorator, the actual method is at stack[2]
        # # - Otherwise, it's at stack[1]
        # if "wrapper" in stack[1].function:
        #     caller_frame = stack[2]  # Skip the decorator's wrapper
        # else:
        #     caller_frame = stack[1]  # Direct call from the method
        # # filename = os.path.splitext(os.path.basename(caller_frame.filename))[0]  # Remove .py extension
        # filename = os.path.basename(caller_frame.filename)

        stack = inspect.stack()
        caller_frame = None

        for frame in stack[1:]:  # Start from stack[1] to skip printl itself
            filename = os.path.splitext(os.path.basename(frame.filename))[0]  # Remove .py extension

            # If it's a decorator wrapper, skip it and move to the next frame
            if frame.function == "wrapper":
                continue

                # If it's not in SKIP_FILES, use this as the actual caller
            if filename not in self.SKIP_FILES:
                caller_frame = frame
                break  # Stop at the first valid caller

        # Fallback: If no valid caller is found, use the last frame (shouldn't happen in normal cases)
        if caller_frame is None:
            caller_frame = stack[1]

        # filename = os.path.splitext(os.path.basename(caller_frame.filename))[0]
        filename = os.path.basename(caller_frame.filename)

        current_timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]: ")
        sn = f"{current_timestamp} {filename}. {str(s)}\n"

        # with open(self.temp, 'a') as f:
        #     f.write(sn)
        with open(self.persistent_log, 'a') as f:
            f.write(sn)
        print(sn)
        try:
            # on Telegram '_' is a special character for italic
            t.send(s.replace('_', '-'))
        except Exception as e:
            print(f"ERROR: Impossible sending message on telegram. {e}.")

    def printK(self, index, K, s):
        if index % K == 0:
            self.printl(s)

    def printTemp(self, s):
        if isinstance(s, str):
            sn = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]: ") + s + "\n"
            with open(self.temp, 'a') as f:
                f.write(sn)
        else:
            print("ERROR: you must specify a str parameter")