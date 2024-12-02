library(shiny)
library(lubridate)
library(ggplot2)
library(forecast)
library(quantmod)
library(TTR)

# Définir l'interface utilisateur
ui <- fluidPage(
  titlePanel("Analyse de Séries Temporelles - Yahoo Finance"),
  sidebarLayout(
    sidebarPanel(
      textInput("sym", label = h6("Entrer le symbole de l'organisation"), value = "HDFCBANK.NS"),
      dateRangeInput("dates", label = h6("Sélectionnez la plage de dates"), 
                     start = "2023-01-01", end = "2024-01-01"),
      sliderInput("split", label = h6("Pourcentage d'entraînement (%)"), min = 80, max = 95, value = 90),
      sliderInput("horizon", label = h6("Horizon de prédiction"), min = 5, max = 10, value = 5),
      h5("Paramètres ARIMA"),
      sliderInput("parp", label = "Autoregression (p)", min = 0, max = 5, value = 2),
      sliderInput("pard", label = "Intégration (d)", min = 0, max = 2, value = 1),
      sliderInput("parq", label = "Moyenne mobile (q)", min = 0, max = 5, value = 2),
      h5("Paramètres saisonniers pour SARIMA"),
      sliderInput("sp", label = "Saison (P)", min = 0, max = 3, value = 1),
      sliderInput("sq", label = "Saison (Q)", min = 0, max = 3, value = 1),
      numericInput("seasonal_period", label = "Période saisonnière", value = 12),
      selectInput("model", label = "Modèle à utiliser", choices = c("ARIMA", "ARIMAX", "SARIMA"), selected = "ARIMA"),
      textInput("exog", label = "Variable explicative (pour ARIMAX)", placeholder = 'Symbole de la variable explicative (ex: SPY)')
    ),
    mainPanel(
      plotOutput("ts_plot"),
      tableOutput("metrics")
    )
  )
)

# Définir le serveur
server <- function(input, output) {
  
  # Fonction pour obtenir les données
  get_data <- reactive({
    symbol <- input$sym
    start_date <- as.Date(input$dates[1])
    end_date <- as.Date(input$dates[2])
    data <- getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
    Cl(data)  # Utiliser les prix de clôture ajustés
  })
  
  # Diviser les données en ensembles d'entraînement et de test
  split_data <- reactive({
    data <- get_data()
    n <- length(data)
    split_point <- floor(n * (input$split / 100))
    train_data <- data[1:split_point]
    test_data <- data[(split_point + 1):n]
    list(train = train_data, test = test_data)
  })
  
  # Graphique de la série temporelle
  output$ts_plot <- renderPlot({
    data <- get_data()
    train_data <- split_data()$train
    test_data <- split_data()$test
    plot(data, type = "l", main = "Série Temporelle", xlab = "Temps", ylab = "Prix ajusté")
    lines(test_data, col = "red", lty = 2)  # Ajouter les données de test
  })
  
  # Métriques d'évaluation
  output$metrics <- renderTable({
    train <- split_data()$train
    test <- split_data()$test
    pp <- input$parp
    pd <- input$pard
    pq <- input$parq
    fit <- arima(train, order = c(pp, pd, pq))
    pred <- forecast(fit, h = length(test))
    mae <- mean(abs(test - pred$mean))
    rmse <- sqrt(mean((test - pred$mean)^2))
    data.frame(MAE = mae, RMSE = rmse)
  })
}

# Lancer l'application
shinyApp(ui = ui, server = server)