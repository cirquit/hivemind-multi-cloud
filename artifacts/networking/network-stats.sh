#!/bin/bash

LOCAL_HOSTNAME=$1
IPERF_TIME_S=10
PING_COUNT=5
PORT=45555
THREADS=1

FILENAME="iperf-profile-$LOCAL_HOSTNAME.csv"

# create csv file header if it does not exist
if [ ! -f $FILENAME ]
then
  echo "timestamp,local_hostname,target_hostname,target_ip,bandwidth,bandwidth_metric,avg_ping_ms" > $FILENAME
fi 

declare -a targets=("34.29.177.183 gc-t4-1-us"
                    "34.140.26.231 gc-t4-2-eu"
                   )

for count in {1..5}:
do
    for i in "${targets[@]}"
    do
        set -- $i 
        TARGET=$1
        TARGET_HOSTNAME=$2
        NOW=`date +"%Y-%m-%d_%H-%M-%S"`
        # get the average time in ms from the last line of ping output
        PING_AVG=`ping $TARGET -c $PING_COUNT | awk -F/ 'END{print $5}'`

        iperf -c $1 -t $IPERF_TIME_S -p $PORT -P $THREADS | \
            awk -vtarget="$TARGET" \
                -vhostname="$TARGET_HOSTNAME" \
                -vlocalhostname="$LOCAL_HOSTNAME" \
                -vavg_ping_ms="$PING_AVG" \
                -vnow="$NOW" \
                'END{print now","localhostname","hostname","target","$7","$8","avg_ping_ms}' >> $FILENAME
    done
done

