Log detection results to Google Sheets
======================================

The autologger publishes one summary row per scan and detection result to a
Google Sheet. Use it after streak or region detection when operators need a
shared, continuously updated view of hit rates during an experiment.

The workflow is:

.. code-block:: text

   detection jobs -> hit metadata -> one logging job -> Google Sheet

Each detection artifact records ``n_detections`` in the metadata for every hit
frame and stores its total processed-frame count as an attribute. The logging
job calculates scan-level statistics from the supplied artifacts and updates
the Sheet. A completed scan with no hits is therefore logged with zeros rather
than being mistaken for a missing result.

Prerequisites
-------------

Complete the normal detection setup first. In particular, the scan
configuration, detector parameters, background metadata, and SLURM script
specification must already work for :ref:`workflows-slurm`.

Install the Google clients into the Conda environment used by the logging job:

.. code-block:: console

   conda install -c conda-forge google-api-python-client google-auth

These Python packages do not provide the ``gcloud`` command. The initial
user-account sign-in uses the separate `Google Cloud CLI
<https://docs.cloud.google.com/sdk/docs/install-sdk>`__ to create Application
Default Credentials. The autologger itself does not invoke ``gcloud``. To
install the CLI with Conda, run:

.. code-block:: console

   conda install -c conda-forge google-cloud-sdk

The spreadsheet is normally created and owned by a user's Google account. The
logger can authenticate either as that user or as a service account with which
the user has shared the spreadsheet. Both routes use Application Default
Credentials (ADC). The optional ``credentials_file`` setting stores only a
path; the credential itself remains separate from ``sheets.json``.

Sign in with the owner's user account
-------------------------------------

Use user credentials when the person who owns the spreadsheet also submits the
logging job:

1. In a Google Cloud project, enable the `Google Sheets API
   <https://console.cloud.google.com/apis/library/sheets.googleapis.com>`__.
2. Configure the project's OAuth consent screen. If the application is in
   external testing mode, add the spreadsheet owner's account as a test user.
3. Follow Google's `OAuth client instructions
   <https://developers.google.com/workspace/guides/create-credentials#desktop-app>`__
   to create a **Desktop app** client ID, then download its JSON file.
4. On the login host, run:

   .. code-block:: console

      unset GOOGLE_APPLICATION_CREDENTIALS
      gcloud_cloud_scope=https://www.googleapis.com/auth/cloud-platform
      gcloud_sheets_scope=https://www.googleapis.com/auth/spreadsheets
      gcloud auth application-default login \
          --client-id-file=/path/to/oauth-client.json \
          --scopes="${gcloud_cloud_scope},${gcloud_sheets_scope}"

5. Sign in as the spreadsheet owner in the browser and approve the requested
   Sheets access. On a cluster host without a browser, add ``--no-browser`` to
   use remote bootstrap. Copy the complete ``gcloud auth application-default
   login --remote-bootstrap=...`` command to a trusted machine that has a
   browser and ``gcloud`` 372.0.0 or newer. Run it there, then paste its output
   into the cluster prompt. Current ``gcloud`` versions do not support
   ``--no-launch-browser`` together with ``--client-id-file``. The second
   ``gcloud`` installation is needed only for this one-time bootstrap exchange,
   not by the autologger.

This is an ADC login, which is separate from ``gcloud auth login``. On Linux,
the resulting credential is normally stored at
``~/.config/gcloud/application_default_credentials.json``. It contains a user
refresh token, so keep it private and do not commit or share it. After this file
has been created, the logger needs the file and the Python clients, but not the
``gcloud`` executable. Revoke the credential when it is no longer needed with:

.. code-block:: console

   gcloud auth application-default revoke

The owner already has permission to edit their spreadsheet, so no additional
sharing step is needed when the logger authenticates as that same account.

Give another identity access to the sheet
-----------------------------------------

Share the user-owned spreadsheet when the logger authenticates as another
Google user or as a service account:

1. Open the spreadsheet while signed in as its owner.
2. Select **Share**.
3. Enter the other user's email address, or the service account's
   ``client_email`` from its JSON key.
4. Select **Editor**, then select **Send**.

Directly sharing the spreadsheet is sufficient for a service account; Google
Workspace domain-wide delegation is not required. If an organisation blocks
sharing outside its domain, ask its Workspace administrator to allow the
service-account address or use the owner's user credentials instead.

For an unattended shared account, create a service account and JSON key using
Google's `service-account instructions
<https://developers.google.com/workspace/guides/create-credentials#service-account>`__.
Store the key outside the repository at a location readable from the SLURM
compute nodes and restrict its permissions, for example:

.. code-block:: console

   chmod 600 /gpfs/cfel/credentials/cbc-autologger.json

Identify the spreadsheet and worksheet
--------------------------------------

For a spreadsheet URL such as:

.. code-block:: text

   https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOp/edit#gid=123456

the spreadsheet ID is the text between ``/d/`` and ``/edit``:
``1AbCdEfGhIjKlMnOp``. The worksheet is the exact name shown on its tab, for
example ``CBC autolog``. The numeric ``gid`` is not used.

Create ``experiments/exfel/sheets.json``:

.. code-block:: json

   {
       "parameters": {
           "spreadsheet_id": "1AbCdEfGhIjKlMnOp",
           "worksheet": "CBC autolog",
           "sort_rows": true,
           "credentials_file": null
       }
   }

Set ``credentials_file`` to an absolute path when this logger should use a
specific user ADC file or service-account key instead of normal ADC discovery:

.. code-block:: json

   "credentials_file": "/gpfs/cfel/credentials/cbc-autologger.json"

Only the path belongs in ``sheets.json``. Keep the referenced credential file
outside the repository, restrict it to its owner (for example with
``chmod 600``), and ensure that the path is accessible on the SLURM compute
nodes. When ``credentials_file`` is ``null`` or omitted, the logger uses normal
ADC discovery.

Use an empty worksheet for the first run. The logger creates its header in
columns A--N. If the worksheet already has a header, it must exactly match the
autologger schema; this prevents unrelated data from being overwritten.

Make credentials available to SLURM
-----------------------------------

For user authentication, do not set ``GOOGLE_APPLICATION_CREDENTIALS``. The
logger discovers the ADC file created by ``gcloud auth application-default
login``. The submitting user's home directory, including ``~/.config/gcloud``,
must be available with the same path on the SLURM compute node.

For service-account authentication, either set ``credentials_file`` in
``sheets.json`` as shown above or set ``GOOGLE_APPLICATION_CREDENTIALS`` in the
script specification. The latter keeps machine-specific authentication out of
the experiment parameters. The relevant part of
``experiments/exfel/script_spec.json`` is:

.. code-block:: json

   {
       "parameters": {
           "partition": "upex",
           "time": "00:10:00",
           "nodes": 1,
           "mem": "2G",
           "chdir": "/gpfs/cfel/user/you/cbclib_v2",
           "output": "experiments/exfel/logs/slurm-%j.out",
           "error": "experiments/exfel/logs/slurm-%j.out",
           "conda_env": "cbc",
           "conda_source": "~/miniforge3/etc/profile.d/conda.sh",
           "define_macros": {
               "GOOGLE_APPLICATION_CREDENTIALS":
                   "/gpfs/cfel/credentials/cbc-autologger.json"
           }
       }
   }

For an interactive command using the service account, export the same variable
in the current shell:

.. code-block:: console

   export GOOGLE_APPLICATION_CREDENTIALS=/gpfs/cfel/credentials/cbc-autologger.json

Run detection and logging
-------------------------

Submit the chunked detection array first. Wait for every array task to finish
before submitting the logging job:

.. code-block:: python

   from cbclib_v2.slurm import Scripts, SLURMJobManager

   manager = SLURMJobManager()

   detect_script = Scripts.sbatch_array.detect(
       373,
       8,
       'streaks',
       'experiments/exfel/scan.json',
       'experiments/exfel/detect_streaks.json',
       'experiments/exfel/script_spec.json',
       out_suffix='online',
   )
   detection_jobs = manager.submit_array(detect_script, wait=False)
   manager.wait_all(detection_jobs)

   log_script = Scripts.sbatch.log(
       373,
       'streaks',
       'experiments/exfel/scan.json',
       'experiments/exfel/sheets.json',
       'experiments/exfel/script_spec.json',
       in_suffix='online',
       sample='lysozyme',
       notes='alignment check',
   )
   manager.submit(log_script)

The result suffix must match in both calls. It distinguishes multiple detection
runs for the same scan. Replace ``streaks`` with ``regions`` and use the region
detector parameters to log region detection.

If detection is already complete, run the logger directly:

.. code-block:: console

   cbclib_cli log 373 streaks experiments/exfel/scan.json \
       experiments/exfel/sheets.json --in-suffix online \
       --sample lysozyme --notes "alignment check"

To log several completed scans, submit them as one job:

.. code-block:: python

   log_script = Scripts.sbatch.log(
       [373, 374, 375],
       'streaks',
       'experiments/exfel/scan.json',
       'experiments/exfel/sheets.json',
       'experiments/exfel/script_spec.json',
       in_suffix='online',
       sample='lysozyme',
       notes='alignment check',
   )
   manager.submit(log_script)

The job processes scans sequentially. This avoids concurrent jobs racing to
rewrite the same worksheet.

Understand the resulting rows
-----------------------------

Each row contains the scan number, detection kind, result suffix, number of
processed frames, number of hits, hit rate, total detections in hit frames,
mean and median detections per hit, average streak length in detector pixels,
the hit threshold, optional sample and notes values, and the UTC update time.
A frame is a hit when its detection count is strictly greater than the
configured threshold. The average length is taken across all streaks in all
saved hit frames, so logging requires detection artifacts containing the full
``data`` table rather than ``--frames-only`` output.

The stable row key is ``(scan number, detection kind, result suffix)``. Running
the logger again with the same key replaces that row, so retrying a completed
logging job does not create a duplicate.

With ``sort_rows`` set to ``true``, the logger orders the machine-owned table
by numeric scan number, then detection kind and result suffix. This places a
new scan at its natural location even when scans are processed out of order.
Set ``sort_rows`` to ``false`` to retain the existing order and append new keys
at the end.

Troubleshooting
---------------

``DefaultCredentialsError``
   With user authentication, confirm that the ADC login file exists in the
   submitting user's home directory and is visible on the compute node. With a
   service account, confirm that ``GOOGLE_APPLICATION_CREDENTIALS`` points to a
   readable key file.

HTTP 403 or permission denied
   Confirm that the Sheets API is enabled. Check that the ADC login used the
   spreadsheet owner's account, or that the owner shared the spreadsheet with
   the authenticated user or service account as an Editor.

User login opens the wrong Google account
   Run ``gcloud auth application-default revoke``, repeat the ADC login, and
   select the spreadsheet owner's account in the browser. A normal
   ``gcloud auth login`` does not replace the ADC identity used by the logger.

Worksheet not found
   Use the exact worksheet tab name in ``worksheet``. Do not use the ``gid``.

Header does not match
   Select an empty worksheet, or restore the autologger's A--N header. The
   logger deliberately refuses to replace a worksheet with a different schema.

Missing Google client libraries
   Install ``google-api-python-client`` and ``google-auth`` in the same Conda
   environment selected by the SLURM script specification.

See :doc:`api_scripts` for the API reference for
:class:`~cbclib_v2.slurm.LogDetections`,
:class:`~cbclib_v2.slurm.GoogleSheetsConfig`, and
:class:`~cbclib_v2.slurm.GoogleSheetsLog`.
