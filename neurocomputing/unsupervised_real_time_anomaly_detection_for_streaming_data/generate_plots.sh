#!/bin/bash

set -e -x

# Figure 1
nab-plot realKnownCause/machine_temperature_system_failure.csv --title "Figure 1" --offline --no-xLabel  --yLabel="Temperature" --start="2013-12-11 06:10:00"

# Figure 2
nab-plot realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv --title "Figure 2" --offline --no-xLabel --yLabel="CPU Utilization (Percent)" --start="2014-02-23 00:00:00" --end="2014-02-25 23:59:59"


# Figure 5
FONT_SIZE=14
nab-plot realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv --title "Figure 5a" --offline --no-xLabel --yLabel="CPU Utilization (Percent)" --fontSize=${FONT_SIZE} --width=1000 --height=400
nab-plot realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv --title "Figure 5b" --offline --no-xLabel --value="raw" --fontSize=${FONT_SIZE} --width=1000 --height=400

# Figure 6
FONT_SIZE=16
nab-plot artificialWithAnomaly/art_load_balancer_spikes.csv --title "Figure 6a" --offline --no-xLabel --yLabel="Latency" --fontSize=${FONT_SIZE} --width=1000 --height=300
nab-plot artificialWithAnomaly/art_load_balancer_spikes.csv --title "Figure 6b" --offline --no-xLabel --value="raw" --fontSize=${FONT_SIZE} --width=1000 --height=300
nab-plot artificialWithAnomaly/art_load_balancer_spikes.csv --title "Figure 6c" --offline --no-xLabel --value="likelihood" --fontSize=${FONT_SIZE} --width=1000 --height=300

# Figure 7
FONT_SIZE=24
nab-plot realAdExchange/exchange-4_cpc_results.csv --title "Figure 7a" --offline --no-xLabel --yLabel="Cost Per Click" --fontSize=${FONT_SIZE}
nab-plot artificialNoAnomaly/art_daily_small_noise.csv --title "Figure 7b" --offline --no-xLabel --yLabel="Metric" --fontSize=${FONT_SIZE}
nab-plot realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv --title "Figure 7c" --offline --no-xLabel --yLabel="CPU Utilization (Percent)"
nab-plot realAWSCloudwatch/grok_asg_anomaly.csv --title "Figure 7d" --offline --no-xLabel --yLabel="Autoscaling Group Size" --fontSize=${FONT_SIZE}
nab-plot realTweets/Twitter_volume_FB.csv --title "Figure 7e" --offline --no-xLabel --yLabel="Tweets Referencing Facebook" --fontSize=${FONT_SIZE}
nab-plot realKnownCause/nyc_taxi.csv --title "Figure 7f" --offline --no-xLabel --yLabel="NYC Taxi Demand" --fontSize=${FONT_SIZE}

# Figure 8
nab-plot realKnownCause/machine_temperature_system_failure.csv --title "Figure 8" --offline --no-xLabel --yLabel="Temperature" --windows --start="2013-12-13 05:20:00" --end="2013-12-20 08:50:00"

# Figure 9
python fig9.py
