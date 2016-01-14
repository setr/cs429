Each student has their own private GitHub repository at:  
<https://github.com/iit-cs429/[your-github-id]>

This is where you will submit all assignments.

Your repository should already contain starter code for each assignment. This starter code has been pulled from the assignment repository at <https://github.com/iit-cs429/assignments>.

Throughout the course, I may update the assignments to clarify questions or add content. To ensure you have the latest content, you can run the `update.sh`, which will fetch and merge the content from the assignments repository.

For each assignment, then, you should do the following:

1. Run `./update.sh` to get the latest starter code.

2. Do the homework, adding and modifying files in the assignment directory. **Commit often!**

3. Before the deadline, push all of your changes to GitHub. E.g.:
  ```
  cd a0
  git add *
  git commit -m 'homework completed'
  git push
  ```

4. Double-check that you don't have any outstanding changes to commit:
  ```
  git status
  # On branch master
  nothing to commit, working directory clean
  ```

5. Double-check that everything works, by cloning your repository into a new directory and executing all tests.
  ```
  cd 
  mkdir tmp
  cd tmp
  git clone https://github.com/iit-cs429/[your_iit_id]
  cd [your_iit_id]/a0
  [...run any relevant scripts/tests]
  ```

6. You can also view your code on Github with a web browser to make sure all your code has been submitted.

7. Assignments contain [doctests](https://docs.python.org/2/library/doctest.html). You can run these for a file `foo.py` using `python -m doctest foo.py`. 
8. Typically, each assignment contains a number of methods for you to complete. I recommend tackling these one at a time, debugging and testing, and then moving onto the next method. Implementing everything and then running at the end will likely result in many errors that can be difficult to track down.

