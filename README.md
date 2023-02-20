# uex-git
This repository has been created for practice during the class at the UEx.

# Basic commands

Add the new/modified/deleted files to the repo:
```
git add <filename>
```
Instead of `filename` you can use `.` or `--all` to add all the target files to the repository.

To delete a file from the repo:
```
git rm --cached <filename>
```
The `--cached` option removes the file from the staging area but remains intact inside the working directory.

To commit a change:
```
git commit -m "<comments>"
```

To push a commit from local to remote:
```
git push
```

Check the status of the repo:
```
git status
```

Reverts file changes:
```
git checkout -- <filename>
```
