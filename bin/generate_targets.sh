#!/bin/sh
# NAME
#         generate_targets.sh - collect raw GA Power logs into a single csv
#
# SYNOPSIS
#         generate_targets.sh RAW_DIRECTORY [PATTERN]
#
# DESCRIPTION
#         The raw GA Power log files are gzipped CSVs without headers. The
#         names of the files are seemingly unrelated to the timeseries they
#         contain. This utility decompresses and concatenates the `mb-007`
#         series of logs, sorts the lines, optionally filters lines, and dumps
#         the result to stdout, with an additional header row.
#
# ARGUMENTS
#         RAW_DIRECTORY     A path to the directory containing the raw logs.
#         PATTERN           A regex to filter the lines, using egrep.


# Echo the header row.
# The titles are the device names.

# Index column.
# The name 'reftime' joins with the NAM data.
echo -n 'reftime'

# The next three columns are undocumented.
echo -n ',unknown1'
echo -n ',unknown2'
echo -n ',unknown3'

# Array A
echo -n ',UGA-A-POA-1-IRR'    # POA Irradiance
echo -n ',UGA-A-POA-2-IRR'    # POA Irradiance
echo -n ',UGA-A-POA-3-IRR'    # POA Irradiance
echo -n ',UGA-A-POA-REF-IRR'  # Cell Temp and Irradiance

# Array B
echo -n ',UGA-B-POA-1-IRR'    # POA Irradiance
echo -n ',UGA-B-POA-2-IRR'    # POA Irradiance
echo -n ',UGA-B-POA-3-IRR'    # POA Irradiance
echo -n ',UGA-B-POA-REF-IRR'  # Cell Temp and Irradiance

# Array D
# NOTE: array D comes before array C
echo -n ',UGA-D-POA-1-IRR'    # POA Irradiance
echo -n ',UGA-D-POA-2-IRR'    # POA Irradiance
echo -n ',UGA-D-POA-3-IRR'    # POA Irradiance
echo -n ',UGA-D-POA-REF-IRR'  # Cell Temp and Irradiance

# Array C
# NOTE: array C comes after array D
echo -n ',UGA-C-POA-1-IRR'    # POA Irradiance
echo -n ',UGA-C-POA-2-IRR'    # POA Irradiance
echo -n ',UGA-C-POA-3-IRR'    # POA Irradiance
echo -n ',UGA-C-POA-REF-IRR'  # Cell Temp and Irradiance

# Array E
echo -n ',UGA-E-POA-1-IRR'    # POA Irradiance
echo -n ',UGA-E-POA-2-IRR'    # POA Irradiance
echo -n ',UGA-E-POA-3-IRR'    # POA Irradiance
echo -n ',UGA-E-POA-REF-IRR'  # Cell Temp and Irradiance

# MDAS weather station
echo -n ',UGA-MET01-POA-1-IRR'  # GHI
echo -n ',UGA-MET01-POA-2-IRR'  # GHI

# SOLYS2
echo -n ',UGA-MET02-GHI-IRR'  # GHI
echo -n ',UGA-MET02-DHI-IRR'  # DHI
echo -n ',UGA-MET02-FIR-IRR'  # DLWIR
echo -n ',UGA-MET02-DNI-IRR'  # DNI

# End of header
echo

# Print the contents of the files.
find $1/group-mb-007*.csv -not -size 0 -print0 |  # Find non-empty files.
xargs -0 cat                                   |  # Read them.
grep -e "$2"                                   |  # Filter out the desired lines.
sort                                              # Sort the lines.
