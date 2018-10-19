import matplotlib.pyplot as plt
import multiprocessing
from optparse import OptionParser
import sequence_simulations
import sys


def fig6a(cliArgs, noises):
  argsTpl = cliArgs + " --noise {}"

  return [
    sequence_simulations.parser.parse_args(argsTpl.format(noise).split(" "))[0]
    for noise in noises
  ] + [
    sequence_simulations.parser.parse_args((argsTpl + " --cells 1")
                                           .format(noise).split(" "))[0]
    for noise in noises
  ]


def fig6b(cliArgs, noises):
  argsTpl = cliArgs + " --noise {}"

  return [
    sequence_simulations.parser.parse_args(argsTpl.format(noise).split(" "))[0]
    for noise in noises
    ]



if __name__ == "__main__":

  parser = OptionParser("python %prog noise [noise ...]")
  parser.add_option("--figure",
                    help="Which figure to plot.  Must be 'A' or 'B'.")
  parser.add_option("--passthru",
                    help=("Pass options through to sequence_simulations.py.  "
                          "See `python sequence_simulations.py --help` for "
                          "options"))

  # Parse CLI arguments
  options, args = parser.parse_args(sys.argv[1:])

  if not args:
    print "You must specify at least one 'noise' argument."
    sys.exit(1)

  if options.figure == "A":
    figure = fig6a
  elif options.figure == "B":
    figure = fig6b
  else:
    print "You must specify one of '--figure A' or '--figure B'"
    sys.exit(1)

  # Convert list of str to list of float
  noises = [float(noise) for noise in args]

  # Run simulations in parallel
  pool = multiprocessing.Pool()
  results = pool.map(sequence_simulations.runExperiment1,
                     figure(options.passthru, noises))

  fig = plt.figure()

  ax = fig.add_subplot(111)
  ax.set_xlabel("Sequence Elements")
  ax.set_ylabel("Accuracy")

  # Plot results
  for result in results:
    ax.plot(result, linewidth=2.0)

  # Legend
  if options.figure == "A":
    ax.legend(["HTM Layer", "First Order Model"], loc="lower right")
  elif options.figure == "B":
    ax.legend(["{}% cell death".format(int(noise * 100)) for noise in noises],
              loc="lower right")

  # Horizontal bar at 50%
  ax.plot([0.5 for x in xrange(len(results[0]))], "--")

  # Re-tick axes
  plt.yticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
             ("10%", "20%", "30%", "40%", "50%", "60%"))
  plt.xticks((2000, 4000, 6000, 8000))

  plt.savefig('results/Fig6{}.pdf'.format(options.figure))

  # Show plot
  plt.show()