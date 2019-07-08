#!/bin/sh

set -euf

apollo --debug train --name linear_v1 -o linear_v1.model ./linear_v1.json ./train.csv
apollo --debug train --name linear_v2 -o linear_v2.model ./linear_v2.json ./train.csv
apollo --debug train --name linear_v3 -o linear_v3.model ./linear_v3.json ./train.csv
