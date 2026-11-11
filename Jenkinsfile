// Pipeline de Jenkins: clona, valida estructura, prueba, analiza y construye la imagen.
// Úsalo como "Multibranch Pipeline" para que las condiciones por rama funcionen.

def REQUERIDOS = [
    'mlops_pipeline/Cargar_datos.ipynb',
    'mlops_pipeline/comprension_eda.ipynb',
    'mlops_pipeline/ft_engineering.py',
    'mlops_pipeline/model_training_evaluation.py',
    'mlops_pipeline/model_deploy.py',
    'mlops_pipeline/model_monitoring.py',
    'Base_de_datos.csv',
    'requirements.txt',
    '.gitignore',
    'setup.bat',
    'readme.md'
]

def ejecutar(String comando) {
    if (isUnix()) { sh comando } else { bat comando }
}
def py() { return isUnix() ? 'venv/bin/python' : 'venv\\Scripts\\python' }

pipeline {
    agent any
    options { timestamps(); disableConcurrentBuilds() }

    // Dispara el pipeline al hacer push/merge en GitHub (requiere webhook).
    // Alternativa sin webhook:  pollSCM('H/5 * * * *')
    triggers { githubPush() }

    stages {
        stage('Clonar repositorio') {
            steps { checkout scm }
        }

        stage('Validar estructura de carpetas') {
            steps {
                script {
                    def faltantes = REQUERIDOS.findAll { !fileExists(it) }
                    if (faltantes) {
                        error "Estructura inválida. Faltan: ${faltantes.join(', ')}"
                    }
                    echo 'Estructura de carpetas correcta'
                }
            }
        }

        stage('Instalar dependencias') {
            steps {
                script {
                    ejecutar(isUnix() ? 'python3 -m venv venv' : 'python -m venv venv')
                    ejecutar("${py()} -m pip install --upgrade pip")
                    ejecutar("${py()} -m pip install -r requirements.txt")
                }
            }
        }

        stage('Pruebas y cobertura') {
            steps {
                script {
                    ejecutar("${py()} -m pytest tests --cov=mlops_pipeline --cov-report=xml --junitxml=test-results.xml")
                }
            }
            post { always { junit allowEmptyResults: true, testResults: 'test-results.xml' } }
        }

        stage('Análisis SonarCloud') {
            // Requiere sonar-scanner instalado y una credencial tipo "Secret text" con id 'sonar-token'
            when { expression { return env.SONAR_ENABLED == 'true' } }
            steps {
                withCredentials([string(credentialsId: 'sonar-token', variable: 'SONAR_TOKEN')]) {
                    script { ejecutar('sonar-scanner') }
                }
            }
        }

        stage('Construir imagen Docker') {
            when { anyOf { branch 'master'; branch 'certification' } }
            steps {
                script { ejecutar("docker build -t creditos-api:${env.BUILD_NUMBER} .") }
            }
        }
    }

    post {
        success { script { notificar('EXITOSO') } }
        failure { script { notificar('FALLIDO') } }
    }
}

// Notificación: correo si se define NOTIFY_EMAIL (requiere configurar SMTP en Jenkins); si no, solo el log.
def notificar(String estado) {
    def asunto = "[Jenkins] ${env.JOB_NAME} #${env.BUILD_NUMBER}: ${estado}"
    def cuerpo = "El pipeline terminó con estado ${estado}.\nDetalle: ${env.BUILD_URL}"
    if (env.NOTIFY_EMAIL) {
        mail to: env.NOTIFY_EMAIL, subject: asunto, body: cuerpo
    } else {
        echo "${asunto}\n${cuerpo}"
    }
}
