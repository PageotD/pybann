pipeline {
  agent any
  stages {
    stage('test') {
      steps {
        sh '''pip install poetry
poetry install'''
      }
    }

  }
}