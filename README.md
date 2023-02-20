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

Move a file:
```
git mv <source> <destination>
```

Show commit logs:
```
git log
```
Common flags: `-p <num>` to limit the output, `--stat`, `-S <function_name>`, `--since`, `--pretty`.

Other options:
```
git reset, restore, revert
```

# Working with branches

Create a new branch and switch to it:
```
git branch <branch_name>
git checkout <branch_name>
```

Shorthand:
```
git checkout -b <branch_name>
```

Switch back to your master branch (**important**):
```
git checkout master
```

Basic merging (after commit changes) in the master:
```
git merge <branch_name>
```

Now that your work is merged in, you have no further need for the branch. You can delete the branch:
```
git branch -d <branch_name>
```

Result:
![basic-merging-2](https://user-images.githubusercontent.com/15891153/220135270-3fcb5c07-af16-4851-96fa-4b9fb8eadd33.png)

# Solve merge conflicts
