#/bin/sh

datasetf=$1
echo "Dataset folder: $datasetf"

spmv=$2
echo "SparseMatrixDenseVector excutable: $spmv"

# Get some unique data for the experiment ID
now=$(date -Iminutes)
hsh=$(git rev-parse HEAD)
exID="$hsh-$now"

# make a folder for results
mkdir -p "results-$exID"

for f in $(cat $datasetf/datasets.txt);
do
	echo "matrix: $f"
	echo "global: $global"
	echo "local: $local"
	echo "Resultfile: result_$f-$global-$l.txt"

	$spmv $datasetf/$f/$f.mtx $f $HOST $exID &> results-$exID/result_$f.txt
	$spmv --experimentId $exID --load-kernels --loadOutput .gold/spmv-$f.gold -g $global -l $l -i 20 -t 25 --all --check $datasetf/$f/$f.mtx &>                  results-$exID/result_$f-$global-$l.txt
done
