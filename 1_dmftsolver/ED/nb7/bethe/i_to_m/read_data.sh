#!/usr/bin/env bash
UINIT=1.5
UFIN=4.0
dU=0.01
N=$(echo "($UFIN - $UINIT)/$dU" | bc)
FILENAME='double_occ.out'
decimal_cutting () {
	if [ ${1:2:4} == '00' ]; then
	    CFLOAT=$(echo $1 | sed -e 's/.00/.0/g')
	elif [ ${1:3:4} == '0' ]; then
	    CFLOAT=${1:0:3}
	else
	    CFLOAT=$1
	fi
}

touch $FILENAME
echo "# U  docc" >> $FILENAME

for ((i=0; i<$(($N+1)); ++i)); do
	U=$(echo "$UINIT + $i*$dU" |  bc -l)
	decimal_cutting $U
	if [ -e result_$CFLOAT ]; then
	    INFO=($(cat result_$CFLOAT))
	    echo "$U ${INFO[3]}" >> $FILENAME
	fi
done
