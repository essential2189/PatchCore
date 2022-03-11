#!/bin/bash
PID=`ps -ef | grep "run2.py" | grep -v grep | awk '{print $2}'`
kill -9 $PID
