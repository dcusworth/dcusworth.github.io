library(shiny)

ui <- fluidPage(
	pageWithSidebar(
	headerPanel('Inputs for CinemaScore prediction'),
	sidebarPanel(
		numericInput('n', 'Number of critics to consider', 15, min = 1, max = 20),
		numericInput('t', 'Minimum number of common films for model fit', 50, min = 10, max = 1000),
		actionButton("submit", "Run prediction")
	),
	mainPanel(
	tabsetPanel(
		tabPanel("Plot", plotOutput('plot1')),
		tabPanel("Text", textOutput("text1")))
	)
	)
)

# Define server logic required to draw a histogram ----
server <- function(input, output, session) {


observeEvent(
eventExpr=input$submit,
handlerExpr={

	sData = c(input$n, input$t)
	#Run model
	dir_line = paste("python", 
		"dynamic_prediction_model/dynamic_prediction.py -n", 
		sData[1], 
		"-t", 
		sData[2])

	withProgress(message="Running XGBoost model"
	system("source activate /home/dcusworth/.conda/envs/py36")
	system(dir_line)
	system("source deactivate")

	#Get results
	dat = read.table("dynamic_prediction_model/results/individual_predictions.csv", sep=",", header=T)
	text_results = readLines("dynamic_prediction_model/results/Prediction_Results.txt") 
	output$text1 = renderText({text_results})

	#Plot results
	output$plot1 <- renderPlot({
		input$submit
		par(mar = c(5.1, 4.1, 0, 1))
		hist(dat$pred)
		abline(v=mean(dat$baseline,na.rm=T), lty=2)
	  })


}

)

}

shinyApp(ui = ui, server = server)



