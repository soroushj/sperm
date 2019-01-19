#!/usr/bin/env bash

declare -a keys=("all" "acrosome" "head" "vacuole")

for k in "${keys[@]}"
do
  if [ "$k" == "all" ]; then
    n=6
  else
    n=2
  fi

# acc
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/acc-$k.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Accuracy';
set yrange [0:1];
set key right bottom;
plot for [i=1:$n] 'saved-results/results/acc-$k.csv' using 0:i smooth bezier;
"

# loss
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/loss-$k.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Loss';
set yrange [0:1];
plot for [i=1:$n] 'saved-results/results/loss-$k.csv' using 0:i smooth bezier;
"

# acc-no-aug
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/acc-$k-no-aug.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Accuracy';
set yrange [0:1];
set key right bottom;
plot for [i=1:$n] 'saved-results/results/acc-$k-no-aug.csv' using 0:i smooth bezier;
"

# loss-no-aug
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/loss-$k-no-aug.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Loss';
set yrange [0:4];
plot for [i=1:$n] 'saved-results/results/loss-$k-no-aug.csv' using 0:i smooth bezier;
"

done
