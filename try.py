# This script requires Python 3.6+ and ibllib 1.4.7+

# Run this script in the directory "ibl-behavior-data-Dec19/" that contains
# one subfolder per lab:
#
#    ~/ibl-behavior-data-Dec19/ $ python one_example.py
#


# We start by importing the ONE light interface.
try:
    from oneibl.onelight import ONE
except ImportError:
    print("Error: Please install ibllib 1.4.7+ with `pip install ibllib`")
    exit()

one = ONE()

# Search all sessions that have these dataset types.
eids = one.search(['_ibl_trials.*'])

# Select the first session.
eid = eids[0]
print(f"Loading session {eid}.")

# List all dataset types available in that session.
dset_types = one.list(eid)
print(f"Available dataset types: {', '.join(dset_types)}")

# Loading a single dataset.
print(f"Loading {dset_types[0]}")
choice = one.load_dataset(eid, dset_types[0])
print(choice)

# Loading an object.
print("Loading the _ibl_trials object.")
obj = one.load_object(eid, "_ibl_trials")
for key, value in obj.items():
    print(key)
    print(value)
    print()

print("The example script ran successfully!")
