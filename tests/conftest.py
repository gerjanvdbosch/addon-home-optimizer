import os
import time

# The tests are written in the house's time zone (sleep hours, local price
# slots, the clock change), as the add-on runs in the one Home Assistant sets;
# CI runners default to UTC. Set before any test module imports
# domain.time, whose LOCAL_TIMEZONE reads the system's zone once.
os.environ["TZ"] = "Europe/Amsterdam"
time.tzset()
