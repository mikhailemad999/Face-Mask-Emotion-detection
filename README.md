# 🧠 Face Mask & Emotion Detection — FaceGuard.AI

A production-grade AI system for real-time monitoring of **face mask compliance** and **human emotion recognition**, built as a graduation project following a strict **8-Step Data Preprocessing Methodology**.

The system utilizes custom deep learning models deployed in a full-stack containerized architecture, combining relative-path routing, real-time WebRTC/Canvas video overlay rendering, and dual-database persistence.

---

## 🚀 Quick Start (Docker Compose)

The entire system, including databases, backend, frontend, celery workers, and GPU passthrough, can be launched with a single command.

### 📋 Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (configured with WSL 2 on Windows)
* **GPU Inference (Optional)**: [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) + Nvidia Drivers (RTX 2060 or newer recommended)

### 🛠️ Execution Steps
1. **Copy the Environment Template:**
   ```bash
   copy .env.example .env
   ```
2. **Build and Start Services:**
   ```bash
   docker compose up --build -d
   ```
3. **Verify running containers:**
   ```bash
   docker compose ps
   ```

### 🔗 Access Endpoints
* **Frontend Web App:** [http://localhost/](http://localhost/)
* **Backend Django REST API:** [http://localhost:8000/api/](http://localhost:8000/api/)
* **Interactive API Documentation (Swagger):** [http://localhost:8000/api/schema/swagger-ui/](http://localhost:8000/api/schema/swagger-ui/)
* **Django Admin Panel:** [http://localhost:8000/admin/](http://localhost:8000/admin/)

---

## 🛠️ Local Development Setup

If you prefer to run services individually for debugging, follow these steps.

### 1. Spin up Databases (via Docker Compose)
Avoid installing databases locally by running only the database containers:
```bash
docker compose up -d db mongo redis
```
This starts:
* **SQL Server 2022** on `localhost:1433`
* **MongoDB 7.0** on `localhost:27017`
* **Redis 7.2** on `localhost:6379`

### 2. Set Up the Environment
Create a `.env` file in the root folder of the project. Python Decouple will recursively search parent directories to read this configuration.
```bash
copy .env.example .env
```
*(Optionally, customize passwords, database names, or toggle `DEBUG=True` inside `.env`).*

---

### 3. Run the Django Backend
Make sure you have **Python 3.11** installed.

> [!IMPORTANT]
> To connect to SQL Server locally, you must install the **Microsoft ODBC Driver 17 for SQL Server** on your host machine.
> * [Download for Windows](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
> * [Download for Linux/macOS](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)

1. **Navigate to the backend directory & create a virtual environment:**
   ```bash
   cd project/backend
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   * **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   * **Windows (CMD):** `.\venv\Scripts\activate.bat`
   * **Linux/macOS:** `source venv/bin/activate`
3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Prepare Database Migrations:**
   ```bash
   python manage.py makemigrations detection
   python manage.py migrate
   ```
5. **Run the ASGI Daphne server:**
   ```bash
   python manage.py runserver 8000
   ```
   *(Alternatively, Daphne can be run explicitly via: `daphne -b 127.0.0.1 -p 8000 config.asgi:application`)*

---

### 4. Run the React Frontend
Ensure you have **Node.js v18 or v20** installed.

1. **Navigate to the frontend directory:**
   ```bash
   cd project/frontend
   ```
2. **Install Node Packages:**
   ```bash
   npm install
   ```
3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
4. **Open in browser:**
   Open [http://localhost:5173/](http://localhost:5173/) to interact with the frontend.

---

### 5. Run the Celery Worker (Optional)
Celery handles asynchronous tasks, such as offline prediction report compiles or image processing queues:
```bash
cd project/backend
# Activate virtualenv first!
celery -A config worker --loglevel=info --concurrency=2
```

---

## 📊 Machine Learning Pipeline (Training & EDA)

The project includes training datasets located in the root directories (`happy`, `sad`, `with_mask`, etc.) and preprocessing/training pipelines in `project/ml`.

### ML Environment Setup
To explore dataset characteristics or run model training:
1. **Navigate to ML directory & install dependencies:**
   ```bash
   cd project/ml
   pip install -r requirements_ml.txt
   ```
2. **Run Pipeline Stages:**
   * **Stage 1 (EDA / Duplicate Hashing):**
     ```bash
     python notebooks/01_explore_duplicates.py
     ```
   * **Stage 2 (Missing Values & Outliers filtering):**
     ```bash
     python notebooks/02_missing_outliers.py
     ```
   * **Stage 3 (Data Visualizations & Class Balancing):**
     ```bash
     python notebooks/03_visualization_balance.py
     ```
   * **Stage 4 (Train Face Mask Detection - PyTorch & ONNX Export):**
     ```bash
     python notebooks/04_train_mask.py
     ```
   * **Stage 5 (Train Emotion Classifier - PyTorch):**
     ```bash
     python notebooks/05_train_emotion.py
     ```

---

## 🏗️ System Architecture

```
               +-----------------------------+
               |    React Frontend Client    |
               |       (Vite dev server)     |
               +--------------+--------------+
                              |
                     /api relative proxy
                              |
                              v
               +--------------+--------------+
               |     Daphne ASGI Backend     |
               |         (Django REST)       |
               +-------+------+-------+------+
                       |      |       |
      +----------------+      |       +-----------------+
      |                       |                         |
      v                       v                         v
+-----+-------+        +------+------+            +-----+-------+
| SQL Server  |        |  MongoDB    |            |   Redis     |
|  2022 (DB)  |        | 7.0 (NoSQL) |            |   (Cache)   |
+-------------+        +-------------+            +-------------+
(Structured Audit Logs, (Detailed Predictions,    (Celery Broker,
 Model Registries)       Bboxes, Confidences)      Channel Layers)
```

### Dual-Database Architecture
This project employs a hybrid storage strategy to maintain both audit capability and analytical precision:
1. **SQL Server 2022**: Stores transactional schema-bound logs.
   * *Examples*: Log identity counters, system status logs, core metrics, registered model variants, parameters count, accuracies, and system information.
2. **MongoDB 7.0**: Stores prediction document entries.
   * *Examples*: Cropped bounding-box dimensions, prediction probabilities, confidence arrays, frame rate metrics, and raw webcam session indicators.

---

## 📁 Repository Structure

```
├── project/
│   ├── backend/             # Django Rest Framework + Channels + Celery code
│   │   ├── config/          # Core Django project settings, ASGI, and Routing
│   │   ├── api/             # API request routing and general middleware
│   │   ├── detection/       # Face mask & emotion models logic and views
│   │   └── analytics/       # Analytics endpoints and log retrieval controllers
│   ├── frontend/            # Vite + React 18 Application code
│   │   ├── src/
│   │   │   ├── pages/       # Dashboard, Live Camera, Analyze, Analytics, Model Registry
│   │   │   └── App.jsx      # App entrypoint and CSS styling imports
│   │   └── package.json     # Node.js dependencies configuration
│   ├── ml/                  # Deep learning pipelines
│   │   ├── models/          # Trained weights (.pt, .onnx, configuration parameters)
│   │   ├── notebooks/       # Scripted files (01-05) for the 8-Step Preprocessing sequence
│   │   └── requirements_ml.txt
│   └── docs/                # Graduation Project Report documentation
├── docker-compose.yml       # Production-ready container configurations
├── .env.example             # Template for local environment config variables
└── README.md                # Project startup documentation (this file)
```

---

## ⚙️ Environment Configuration

| Variable | Default Value | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | `dev-secret-change-me` | Django cryptography token |
| `DEBUG` | `True` | Activates detailed traceback error logs |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Hosts approved for routing |
| `SQL_SERVER_HOST` | `localhost` | Relational SQL Server container ip/host |
| `SQL_SERVER_DB` | `FaceMaskEmotionDB` | Database identifier |
| `SQL_SERVER_USER` | `sa` | SQL Server superuser |
| `SQL_SERVER_PASS` | `YourStrong!Pass123` | Secure database entry password |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection URL |
| `MONGODB_DB` | `face_mask_emotion` | Analytics database identifier |
| `REDIS_URL` | `redis://localhost:6379` | Broker location for Celery & WebSockets |
| `ML_MODELS_DIR` | `../ml/models` | Backend reference directory for trained weights |

---

## 🛟 Troubleshooting

* **Vite Proxy / Daphne CORS Issues:** Ensure `VITE_API_URL` is set to `http://localhost:8000/api` during local setup, and configure Django's `CORS_ALLOWED_ORIGINS` correctly if custom domains are utilized.
* **`npm run dev` Fails with "is not recognized" Error:** If the project path contains an `&` character (e.g., `Face Mask & Emotion detection`), the shell interprets `&` as a command separator and breaks the npm script. Use the direct Node.js command instead: `node .\node_modules\vite\bin\vite.js` from the `project/frontend` directory.
* **SQL Server Driver Errors:** If running local Django on Windows and getting ODBC errors, double-check that **ODBC Driver 17** is selected in your `settings.py` `DATABASES['default']['OPTIONS']['driver']` configurations and installed locally.
* **GPU Training Failures:** PyTorch training requires a CUDA-capable Nvidia card (like RTX 2060) with a correct CUDA toolkit version. If `torch.cuda.is_available()` returns `False`, PyTorch will fallback automatically to CPU training, which runs significantly slower.
