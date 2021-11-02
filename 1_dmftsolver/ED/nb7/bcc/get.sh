#!/usr/bin/env bash
#DIR='./i_to_m'
if [ $# -eq 0 ]; then
   echo "ARG[1]: directory to load files"
   exit 1
fi

DIR=$1
FILEs=($(find $DIR -type f -name 'result*'))

function remove_prefix () {
    local PREFIX_=$(echo $1 | sed -e 's/\//\\\//g')
    RES=$(echo $2 | sed -e "s/$PREFIX_//g")
}

remove_prefix './' $DIR
OFILE=result_${RES}

if ! [ -e $OFILE ]; then
    touch $OFILE
else
    rm $OFILE
fi

for FILE in ${FILEs[@]}; do
    CONT=($(cat $FILE))
    remove_prefix $DIR/result_ $FILE
    U=$RES
    echo "$U ${CONT[4]}" >> $OFILE
done

sort -k1 -n $OFILE > $OFILE.dat
rm $OFILE
