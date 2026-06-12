Workflows
=========

**cbclib_v2** is designed to fit different working styles.  The same
processing pipeline can be driven interactively from a Jupyter notebook,
from the ``cbclib_cli`` command-line tool, or fully automated across a
compute cluster via SLURM.  All three workflows are built on a single
JSON-based scan configuration.

.. _workflows-scan-config:

Scan configuration
------------------

Every workflow starts from a :class:`~cbclib_v2.scripts.ScanConfig` object that
describes the experiment in full: where the raw data live, how the detector is
arranged, which processing steps to run, and where to write results.  The
configuration is stored as a JSON file and loaded with
:meth:`~cbclib_v2.scripts.ScanConfig.read`:

.. code-block:: python

   from cbclib_v2.scripts import ScanConfig

   config = ScanConfig.read('experiments/exfel/scan.json')
   run    = config.run()   # returns XFELRun or SwissFELRun

Below is an annotated example for a run at the European XFEL (SPB instrument,
run 373):

.. code-block:: json

   {
       "parameters": {
           "scan_num":   373,
           "image_kind": "stacked"
       },
       "data": {
           "facility":       "XFEL",
           "data_dir":       "/gpfs/exfel/exp/SPB/202302/p004456/proc/r{0:04d}",
           "hdf5_protocol":  "experiments/exfel/agipd_protocol.json",
           "file_pattern":   "CORR-R{0:04d}-JNGFR{1:02d}-S(\\d{5})\\.h5",
           "geometry_file":  "/gpfs/exfel/exp/.../jungfrau_4456_v1.geom",
           "num_modules":    8,
           "starts_at":      1
       },
       "metadata": {
           "n_frames":   25,
           "output_dir": "/gpfs/exfel/exp/.../usr/metadata"
       },
       "metalist": {
           "n_frames":  21,
           "spacing":   40,
           "output_dir": "/gpfs/exfel/exp/.../usr/metalist"
       },
       "detect": {
           "hit_threshold": 15,
           "streaks_dir":   "/gpfs/exfel/exp/.../usr/streaks",
           "regions_dir":   "/gpfs/exfel/exp/.../usr/regions"
       },
       "setup": {
           "setup_file": "experiments/exfel/geometry.json",
           "unit_file":  "experiments/exfel/I3C_unit_cell.json",
           "xtals_dir":  "/gpfs/exfel/exp/.../usr/xtals"
       },
       "system": {
           "platform":    "gpu",
           "cuda_allocator": "default",
           "num_threads": 0
       }
   }

**Key fields:**

``parameters``
   ``scan_num`` — run number passed to :func:`~cbclib_v2.open_run`.
   ``image_kind`` — ``"stacked"`` keeps per-module stacks (fast I/O);
   ``"full"`` assembles them into a single lab-frame image.

``data``
   Facility identity and HDF5 layout.  ``facility`` selects the run class
   (:class:`~cbclib_v2.XFELRun` for ``"XFEL"``, :class:`~cbclib_v2.SwissFELRun`
   for ``"SwissFEL"``, :class:`~cbclib_v2.LCLSRun` for ``"LCLS"``).

``metadata``
   Parameters for the background-estimation step: number of frames averaged
   per background estimate and where to write the result.

``metalist``
   Controls the generation of a *metalist*: a set of background estimates
   sampled at ``n_frames`` points across the dataset (one point every
   ``spacing`` events).  The collection of backgrounds is used to perform a
   PCA-based background decomposition that separates crystal signal from
   diffuse scatter.  See :doc:`background_subtraction` for details.

``detect``
   Hit-finding thresholds and output directories for streaks and region
   detections.  See :doc:`streak_detection` for details on the detection
   parameters.

``setup``
   Crystal geometry and unit-cell files used by the indexer.

``system``
   Backend selection (``"cpu"`` or ``"gpu"``), GPU allocator mode, and
   OpenMP thread count (``0`` = all available cores).  ``cuda_allocator`` may
   be ``"default"`` for the safest backend-native allocator behavior or
   ``"cuda_malloc_async"`` for unified CUDA stream-ordered allocation across
   cbclib, CuPy, and JAX/XLA on compatible GPU nodes.

.. _workflows-notebook:

Interactive exploration (Jupyter notebook)
------------------------------------------

For exploratory analysis or algorithm tuning, the full pipeline can be run
cell by cell in a notebook.  This is the most flexible mode: every
intermediate result is available for visualisation, and parameters can be
tweaked without restarting.

.. note::

   Steps 3-4 below assume the background whitefield has already been computed
   and stored in an HDF5 file.  You can produce it interactively (see
   :doc:`background_subtraction`) or via the CLI (``cbclib_cli metadata``
   described in the :ref:`Command-line scripts <workflows-cli>` section below).

.. code-block:: python

   import numpy as np
   import cbclib_v2 as cbc
   from cbclib_v2.scripts import ScanConfig
   from cbclib_v2.label import Structure
   from cbclib_v2.annotations import NumPy

   # --- 1. Load configuration and open the run ---
   config = ScanConfig.read('experiments/exfel/scan.json')
   run    = config.run()

   cbc.set_cpu_config(64)
   xp = NumPy

   # --- 2. Load raw detector frames ---
   indices  = run.indices()
   frames   = [42, 55, 71]          # pick specific frame numbers
   images   = run.data(indices[frames], geometry=False, xp=xp)

   # --- 3. Subtract background and compute SNR ---
   params   = cbc.scripts.StreakFinderConfig.read('experiments/exfel/detect_streaks.json')
   metadata = params.scaling.metadata(config.find_metadata(0), xp)

   from cbclib_v2.scripts import scale_background
   data = scale_background(frames, images, metadata, params.scaling)
   data = data.update_snr()

   # --- 4. Run the streak-detection pipeline ---
   structure = Structure([0, 0, 3, 3], 4)
   finder    = data.streak_detector(structure, vmin=2.5)

   regions  = finder.detect_regions(npts=25, connectivity=Structure([0, 0, 2, 2], 2))
   labels, peaks    = finder.detect_peaks(regions)
   linelets, labels = finder.fit_linelets(labels, peaks)
   streaks          = finder.detect_streaks(labels.keep_best(0.5), peaks, linelets,
                                         xtol=2.0, nfa=1)
   labeled          = finder.streak_labels(streaks, labels, peaks)
   lines            = finder.line_fit(labeled)
   scores           = finder.min_support(labeled, lines, xtol=2.0)

   detected = finder.to_streaks(lines[scores >= 10.0])
   print(f"Detected {len(detected):d} streaks across {len(frames):d} frames")

.. seealso::

   :doc:`background_subtraction`
      How to produce the SNR frames that serve as input to streak detection.

   :doc:`streak_detection`
      Full description of the streak-detection algorithm and parameters.

.. _workflows-cli:

Command-line scripts
--------------------

The ``cbclib_cli`` entry point exposes the same pipeline as a series of
subcommands.  This is convenient for processing a whole run with a single
shell command or a simple shell script, without writing Python.

The pipeline runs in four sequential steps:

.. code-block:: bash

   # Step 1 — compute the background whitefield
   cbclib_cli metadata experiments/exfel/scan.json \
                        experiments/exfel/metadata.json

   # Step 2 — partition data into per-file chunks
   #   -c <chunk_id>  -n <total_chunks>
   cbclib_cli metalist experiments/exfel/scan.json \
                       experiments/exfel/metadata.json \
                       -c 0 -n 7

   # Step 3 — detect streaks in each chunk (repeat for each chunk)
   cbclib_cli detect streaks experiments/exfel/scan.json \
                              experiments/exfel/detect_streaks.json \
                              -c 0 -n 7

   # Step 4 — merge per-chunk results into a single HDF5 file
   cbclib_cli compile streaks experiments/exfel/scan.json

**Step 1** reads raw frames from the facility HDF5 files and computes the
per-module background whitefields.  The result is written to a single HDF5
file in the ``metadata.output_dir`` specified in ``scan.json``.

**Step 2** *(only needed for PCA-based background subtraction)* — builds a
*metalist*: a collection of background estimates at regularly-spaced points
across the full dataset.  One lightweight HDF5 *metalist* file is written per
chunk; each file records the frame indices and background coefficients for
that portion of the scan.  The ``-c`` / ``-n`` flags select chunk ``c`` out
of ``n`` total.  When a single static background suffices, this step can be
skipped.

**Step 3** reads one metalist (or metadata) file, subtracts the background,
and runs the streak (or region) detector on the resulting SNR frames.  One
result file is written per chunk into ``detect.streaks_dir``.

**Step 4** *(optional)* — concatenates all per-chunk result files into a
single HDF5 file suitable for the indexer.  Skip this step if you prefer to
work with the per-chunk files directly.

Replace ``streaks`` with ``regions`` in steps 3–4 to run the region
detector instead.

.. _workflows-slurm:

SLURM batch pipeline
--------------------

On the MAXWELL cluster (or any SLURM system) all three compute-intensive
steps can be submitted as batch jobs from a notebook or a script.
:class:`~cbclib_v2.slurm.SLURMJobManager` manages job submission and
polling; :class:`~cbclib_v2.slurm.Scripts` builds the ``sbatch`` scripts.

.. code-block:: python

   from cbclib_v2.slurm import Scripts, ScanConfig, SLURMJobManager

   manager = SLURMJobManager()

   config   = ScanConfig.read('experiments/exfel/scan.json')
   run      = config.run()
   n_chunks = 7   # number of file chunks to split the run into for parallel processing

   # --- detect streaks: one SLURM task per file chunk ---
   detect_script = Scripts.sbatch_array.detect(
       'streaks',
       'experiments/exfel/scan.json',
       'experiments/exfel/detect_streaks.json',
       'experiments/exfel/script_spec.json',
       n_chunks,
   )
   job_id = manager.submit_array(detect_script, range(n_chunks), wait=False)

   # --- wait for all tasks to finish ---
   manager.wait_all(job_id)   # block until all tasks finish

   # --- OR: asynchronously read the job's output while it's running ---
   output = manager.get_output(detect_script, job_id[0])
   async for line in manager.stream_job(output, 0.1):
       print(line, end='', flush=True)

   # --- compile results: single job ---
   compile_script = Scripts.sbatch.compile(
       'streaks',
       'experiments/exfel/scan.json',
       'experiments/exfel/script_spec.json',
   )
   manager.submit(compile_script)

The SLURM job parameters (partition, memory, time limit, conda environment)
are read from a *script spec* file:

.. code-block:: json

   {
       "parameters": {
           "partition":     "upex",
           "nodes":         1,
           "mem":           "0",
           "time":          "01:00:00",
           "conda_env":     "cbc",
           "conda_source":  "~/miniforge3/etc/profile.d/conda.sh",
           "chdir":         "/gpfs/cfel/user/you/cbclib_v2",
           "output":        "experiments/exfel/logs/slurm-%j.out",
           "error":         "experiments/exfel/logs/slurm-%j.out",
           "exclusive":     false
       }
   }

The ``submit_array`` call expands to a ``sbatch --array`` command; each
task receives its chunk index via the ``SLURM_ARRAY_TASK_ID`` environment
variable.  :meth:`~cbclib_v2.slurm.SLURMJobManager.wait_all` polls until
all tasks complete and raises on failure.

.. seealso::

   :doc:`api_scripts`
      Full API reference for all scripts, configuration classes, and SLURM
      job-management types.
