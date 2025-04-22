# MLSet

This application, built on FastAPI, allows you to evaluate the capabilities of machine learning using Scikit-learn, TensorFlow, and PyTorch libraries, comparing their performance and accuracy. The application offers a full-fledged user interface with registration and authorization, as well as the ability to display training results on the user page.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes. See the **deployment** section for instructions on how to deploy the project on a live system.

The application is developed using Python 3.11 and the corresponding requirements. It is not guaranteed to work with other versions of Python.

### Prerequisites

To get started, you'll need to have Python 3.11 installed along with `pip`. If you don't have them, please install Python from the [official website](https://www.python.org/downloads/).

### Installing

1. **Clone the repository:**

    First, clone the repository to your local machine:
    
    ```bash
    git clone https://github.com/Art21042147/ML_Set.git
    cd ML_Set
    ```

2. **Create and activate a virtual environment:**

    Create a virtual environment to manage dependencies:
    
    ```bash
    python -m venv .venv
    ```

    Activate the virtual environment:
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

3. **Install required dependencies:**

    Install the necessary Python packages using pip:
    
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the database:**

    To connect to the PostgreSQL database and manage authorization, create a `.env` file in the root of the project and add the following lines:

    ```
    # DB Configuration
    POSTGRES_USER=your_postgres_username
    POSTGRES_PASSWORD=your_postgres_password
    POSTGRES_SERVER=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB_NAME=your_db_name

    # JWT Configuration
    SECRET_KEY=your_secret_key
    ALGORITHM=HS256
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    ```

5. **For Windows systems:**

    It's recommended to add a port configuration in `.env` and `config.py`:
    
    ```
    PORT=8080
    ```

    In the `main.py` file, update the code to use the port from the `.env` file:

    ```python
    from app.core.config import config
    
    port = int(config.PORT) if config.PORT else 8000
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
    ```

6. **Run the application:**

    After setting everything up, you can start the application with:

    ```bash
    uvicorn main:app --reload
    ```

    The application will be available at `http://127.0.0.1:8000` (or the port you specified).

## Project Structure

- **`core:`** Handles secret data from `.env`, provides application settings, and manages user authorization and registration.
- **`db:`** Contains modules for initializing and interacting with the database.
- **`ml:`** The core of the application, where the machine learning modules reside.
- **`routers:`** Manages request processing and routing.
- **`schemas:`** Data validation using Pydantic models.
- **`static:`** Contains the ML logo `.svg` and CSS styles.
- **`templates:`** Contains HTML templates for rendering pages.
- **`main.py:`** The entry point for running the application.
- **`tests:`** unit-tests.

## Built With

* [FastAPI](https://fastapi.tiangolo.com/)
* [Bootstrap](https://getbootstrap.com/docs/5.3/getting-started/introduction/)
* [Jinja2](https://jinja.palletsprojects.com/en/stable/)
* [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)
* [TensorFlow](https://www.tensorflow.org/tutorials)
* [PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Screenshots

### Main Page
The main page of the application provides an interface with a brief description of the purpose of the application, a presentation of the proposed machine learning libraries with links to official resources. To continue working with the application, the user is asked to log in or register.

![Main Page](screenshots/Main_page.png)

---

### Log In
When you click the `Log In` or `Register` buttons, a pop-up window appears in which the user can enter their registration data.

![Log In](screenshots/Log_in.png)

---

### User Page
After registration or authorization, the user page opens with a greeting and further instructions on how to use the application. The main part of the page is occupied by a description of the proposed datasets with links to the official resource. On the right are drop-down lists for selecting learning libraries, datasets and tasks.
The `Start Learning` button starts the learning process according to the selected parameters.

![User Page](screenshots/User_page.png)

---

### Learning Results
The learning results are dynamically displayed on the user page and stored in the `app/ml/predictions` directory.

![Learning Results](screenshots/Learning_result.png)


## Authors

* **Arthur Ilin** - *Core development and implementation* - [Art21042147](https://github.com/Art21042147)
