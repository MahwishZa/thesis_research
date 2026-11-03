# Dataset folder

Extract the skin-lesion dataset ZIP here. Nothing in this folder except this file is committed to Git.

```powershell
Expand-Archive -Path "C:\path\to\your\dataset.zip" -DestinationPath data
```

The program expects a training folder and a testing folder, each with one subfolder per class, for example:

```
data/
  <training folder>/<class name>/<image files>
  <testing folder>/<class name>/<image files>
```

If the ZIP adds an extra top folder, pass it as `--root` (for example `--root "data\<top folder>"`).
The profile command also writes thumbnail sheets to `data/contact_sheets/`; they stay here, out of Git.
