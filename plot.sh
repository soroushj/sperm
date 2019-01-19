#!/usr/bin/env bash

# acc-all
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/acc-all.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Accuracy';
set yrange [0:1];
set key right bottom;
plot for [i=1:6] 'saved-results/results/acc-all.csv' using 0:i smooth bezier;
"

# loss-all
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/loss-all.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Loss';
set yrange [0:1];
plot for [i=1:6] 'saved-results/results/loss-all.csv' using 0:i smooth bezier;
"

# acc-all-no-aug
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/acc-all-no-aug.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Accuracy';
set yrange [0:1];
set key right bottom;
plot for [i=1:6] 'saved-results/results/acc-all-no-aug.csv' using 0:i smooth bezier;
"

# loss-all-no-aug
gnuplot -e "
set terminal pdf;
set output 'saved-results/results-plot/loss-all-no-aug.pdf';
set datafile separator ',';
set key autotitle columnhead;
set xlabel 'Iterations';
set ylabel 'Loss';
set yrange [0:4];
plot for [i=1:6] 'saved-results/results/loss-all-no-aug.csv' using 0:i smooth bezier;
"
