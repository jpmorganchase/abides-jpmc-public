
# Developer Guide


## Development Policies

Internal development is done on Github at [https://github.com/jpmorganchase/abides-jpmc-private]().

### Git Processes

- Development is done on the ``dev`` branch. At certain intervals this code will be merged
  into the ``main`` branch..
- Some small commits can be applied directly to the dev branch. Examples include:
    - Spelling/wording corrections.
    - Small formatting changes.
    - Requirement version bumping.
- Any larger commit that has an effect on code should be developed on a separate branch
  and merged with a pull request.
- Pull requests should have at least one reviewer. This can be assigned on Github.


### Dev Branch Merge Requirements

- Code must be documented:
    - This includes documentation of the actual code and documentation of any functions
      and classes.
    - New classes should be accompanied with a new Sphinx documentation page.

- Unit tests must pass.

- Code must be formatted according the the [black](https://github.com/psf/black)
formatter tool style guidelines.

- Outputs should match:
    - The RMSC03 and RMSC04 tests should produce identical results to the previous commit
      unless explicity stated with the reason for the divergence.

### Prod Branch Merge Requirements

- Commits to the ``prod`` branch should only be merges from the dev ``branch``.
- An exception to this is hotfixes.
- Any breaking changes (of API or outputs) compared to the previous commit should be
  clearly documented and communicated to the team.


## Code Style

- Code style should follow [PEP8](https://www.python.org/dev/peps/pep-0008/).
- Code must also be formatted according to the [black](https://github.com/psf/black)
formatting tool.
- Type annotations should be provided for function signatures and class attribute declarations.

- [isort](https://pypi.org/project/isort/) is suggested for organising imports.

- [pre-commit](https://pre-commit.com/) can be used for automatically applying these changes
when commiting.

## Setting up pre-commit hooks

Git hooks are a tasks that can be run after a commit is created by the user but before it is
confirmed and entered into the git history.

These tasks can potentially stop a commit from being confirmed.

The [pre-commit](https://pre-commit.com/) tool is used to manage these hooks.

The configuration can be found in the `.pre-commit-config.yaml` file in the repository root.

Currently the following hooks are enabled:

- `pytest-check`: checks all unit tests pass before allowing the commit.
- `no-commit-to-branch`: prevents any commits directly to the master branch.


### Commands

To install pre-commit run:
```
$ python -m pip install -r requirements-dev.txt
```

To enable pre-commit hooks run:
```
$ pre-commit install
```

To disable pre-commit hooks run:
```
$ pre-commit uninstall
```

To test the pre-commit hooks without creating a git commit run:
```
$ pre-commit run --all-files
```

## Testing


### Unit Tests

ABIDES uses the pytest framework for unit tests. All tests are contained within
directories named `test` within the respective sub-project directories.

### Regression Testing (Macro-testing)
In order to test the code, it is possible to use [test_current_vs_past_commit](version_testing/test_current_vs_pastcommit.py)
The following steps are happening: 
* Running simulation 
    * ABIDES simulation is run with a commit version of your choice (sha_old)
    * The Order Book and the simulation time are saved
    * ABIDES simulation is run with a commit version of your choice (sha_new) or with your current working tree (sha_new = 'CURRENT')
    * The Order Book and the simulation time are saved
* Comparison of the Order Books
    * If the Order Books are matching, the test is passed 
    * Else the test is failed



## Documentation

- Code should be documented when written.
- Classes and functions should have [Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
  doc strings.
- Non-code documentation should be placed in ``doc`` directory in RST format.

### Writing Documentation Strings

```python
class ExampleClass:
    """A short sentence describing the class should be on the first row.

    Triple quotes are used for documentation comments.

    Each class should have a description describing the class and it's functionality.

    Class attributes that are relevant for users of the class should be given as follows:

    Attributes:
        price: The order limit price.
        quantity: The amount of shares traded.

    Types are not needed here as they will be taken from type annotations.

    Single backticks can be used to highlight code strings. E.g. `ExampleClass`.
    """

    def do_something(self, x: int, y: int) -> bool:
        """Does something with x and y.

        The same style should be used for functions and class methods. Note the type
        annotations.

        Here we document the function/method arguments and return value if needed:

        In class methods we do not need to document the `self` parameter.

        Arguments:
            x: The first number.
            y: The second number.

        Returns:
            True if x > y else False

        """

        return x > y
```

### Building Documentation Pages

To build the documentation pages run the following:

```bash
sphinx-build -M html docs/ docs/_build
```

To view the pages in a web browser on your local machine you can start a Python web server:

``` bash
cd docs/_build/html && python3 -m http.server 8080
```

Then navigate to [http://localhost:8080]() in your web browser.


## Useful Links

- PEP8: Official Python Style Guide:
    [https://www.python.org/dev/peps/pep-0008/]()

- Type annotations cheat sheet:
    [https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html]()

- 'Google Style' Python documentation guide:
    [https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html]()

- Black code formatter:
    [https://black.readthedocs.io/en/stable/]

- Sphinx Doc basics guide:
    [https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html]()
