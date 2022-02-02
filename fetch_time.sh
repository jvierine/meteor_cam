
tstamp="2022_02_02_00_11 2022_02_02_00_12 2022_02_02_03_03 2022_02_02_03_02"
stats="131 132 133"

for s in $stats;
do
    echo $s $t    
    scp -P 22$s ams@localhost:/home/ams/amscams/conf/as6.json as6.json.$s
    for t in $tstamp;
    do
	rsync -av --progress  -e "ssh -p 22$s" "ams@localhost:/mnt/ams2/HD/$t*.mp4" .
    done
done
exit



#rsync -av --progress  -e 'ssh -p 22131' "ams@localhost:/mnt/ams2/HD/2022_02_02_00_12*.mp4" .

#rsync -av --progress  -e 'ssh -p 22132' "ams@localhost:/mnt/ams2/HD/2022_02_02_00_11*.mp4" .
#rsync -av --progress  -e 'ssh -p 22132' "ams@localhost:/mnt/ams2/HD/2022_02_02_00_12*.mp4" .

#rsync -av --progress  -e 'ssh -p 22133' "ams@localhost:/mnt/ams2/HD/2022_02_02_00_11*.mp4" .
#rsync -av --progress  -e 'ssh -p 22133' "ams@localhost:/mnt/ams2/HD/2022_02_02_00_12*.mp4" .


#rsync -av --progress  -e 'ssh -p 22131' "ams@localhost:/mnt/ams2/HD/2022_02_02_03_03*.mp4" .
#rsync -av --progress  -e 'ssh -p 22131' "ams@localhost:/mnt/ams2/HD/2022_02_02_03_02*.mp4" .

#rsync -av --progress  -e 'ssh -p 22132' "ams@localhost:/mnt/ams2/HD/2022_02_02_03_03*.mp4" .
#rsync -av --progress  -e 'ssh -p 22132' "ams@localhost:/mnt/ams2/HD/2022_02_02_03_02*.mp4" .

#rsync -av --progress  -e 'ssh -p 22133' "ams@localhost:/mnt/ams2/HD/2022_02_02_03_03*.mp4" .
#rsync -av --progress  -e 'ssh -p 22133' "ams@localhost:/mnt/ams2/HD/2022_02_02_03_02*.mp4" .

#scp -P 22133 ams@localhost:/home/ams/amscams/conf/as6.conf as6.conf.133
#scp -P 22132 ams@localhost:/home/ams/amscams/conf/as6.conf as6.conf.132
#scp -P 22131 ams@localhost:/home/ams/amscams/conf/as6.conf as6.conf.131
