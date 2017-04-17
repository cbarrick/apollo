for YYYY in 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014; do
    for MM in 01 02 03 04 05 06 07 08 09 10 11 12; do
        for DD in 01 02 03 04 05 06 07 08 09 10 \
                  11 12 13 14 15 16 17 18 19 20 \
                  21 22 23 24 25 26 27 28 29 30 31; do
            for HH in 00 06 12 18; do
                ./get_grib.pl data ${YYYY}${MM}${DD}${HH} 0 0 0 TMP:UGRD:VGRD:RH:TCDC all /mnt/data5tb/grib/NAM-ANL
            done
        done
    done
done
