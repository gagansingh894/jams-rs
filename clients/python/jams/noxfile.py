import nox
import os

nox.options.sessions = ["lint"]

# Define the minimal nox version required to run
nox.options.needs_version = ">= 2024.3.2"

@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"], reuse_venv=True)
def install_requirements(session):
    """Install dependencies from requirements.txt."""
    session.log("Installing dependencies from requirements.txt...")
    session.install("-r", "requirements.txt")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"], reuse_venv=True)
def lint(session):
    """Lint code using mypy and ruff."""
    session.run("make", "lint")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"], reuse_venv=True)
def build_and_check_dists(session):
    session.install("build", "check-manifest==0.43", "twine")
    # If your project uses README.rst, uncomment the following:
    # session.install("readme_renderer")

    session.run("check-manifest", "--ignore", "noxfile.py,tests/**")
    session.run("python", "-m", "build")
    session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"], reuse_venv=True)
def tests(session):
    build_and_check_dists(session)

    generated_files = os.listdir("dist/")
    generated_sdist = os.path.join("dist/", generated_files[1])

    session.install(generated_sdist)

    session.run("py.test", "tests/", *session.posargs)