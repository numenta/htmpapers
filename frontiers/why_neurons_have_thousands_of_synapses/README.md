This directory contains Python scripts that test some specific aspects of
temporal memory.

The script runs sequence simulations with artificial data. The input stream
contains high-order sequences mixed with random elements. The maximum possible
average prediction accuracy of this data stream is 50%. The script is designed to
test the following properties:

1. High order learning - the sequences require high order sequence memory in 
order to achieve perfect scores.

2. Continuous learning - the sequences are switched out in the middle of
training.

3. Fault tolerance - some of the cells in temporal memory are killed in the
middle of training and we observe how the performance changes.  This is
implemented using a special class called `FaultyTemporalMemory` which kills off
a random percentage of cells.


This script was used to generate the graphs in ([Hawkins & Ahmad, 2016][1]). Here 
are the images from that paper:

![](images/figure.png)

Simulation results of the sequence memory network. A) High-order on-line learning.
The red line shows the network learning and achieving maximum possible
performance after about 2500 sequence elements. At element 3000 the repeated
patterns in the data stream were changed. Prediction accuracy drops and then
recovers as the model learns the new temporal structure. For comparison, the
lower performance of a first-order network is shown in blue. B) Robustness of
the network to damage. After the network reached stable performance we
inactivated a random selection of neurons. At up to 40% cell death there is
almost no impact on performance. At greater than 40% cell death the performance
of the network declines but then recovers as the network relearns using
remaining neurons.

Installing HTM Research Library
-------------------------------

All the scripts in this directory depends on Numenta's [HTM Research library][2].
This library requires [python 2.7][3] to be installed and avaialble on Windows, 
Linux and OSX plaforms.

To install [HTM Research library][2] and its dependencies, use the following command:

```
pip install -r requirements.txt
```

Generating plots
----------------

A helper utility generate_plots.py is provided to assist in reproducing figure 
6A-B from ([Hawkins & Ahmad, 2016][1])
### Figure 6A

```
python generate_plots.py 0.0 --figure A --passthru="--name Fig6A"
```

### Figure 6B

```
python generate_plots.py 0.4 0.5 0.6 0.75 --figure B --passthru="--name Fig6B --simulation killer"
```

[1]: http://journal.frontiersin.org/article/10.3389/fncir.2016.00023/full
[2]: https://github.com/numenta/htmresearch
[3]: https://www.python.org/downloads

--------------------------------------------------------------------------------
* J. Hawkins, S. Ahmad, <em>Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex</em>, Front. Neural Circuits. 10 (2016) 1–13. doi:10.3389/fncir.2016.00023.
http://journal.frontiersin.org/article/10.3389/fncir.2016.00023/full