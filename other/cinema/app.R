library(shiny)

ui <- fluidPage(
	pageWithSidebar(
	headerPanel('Inputs'),
	sidebarPanel(
		numericInput('n', 'Number of critics to consider', 15, min = 1, max = 20),
		numericInput('t', 'Minimum number of common films for model fit', 50, min = 10, max = 1000),
		actionButton("submit", "Run prediction")
	),
	mainPanel(
	tabsetPanel(
		tabPanel("Plot", plotOutput('plot1')),
		tabPanel("Text Summary", textOutput("text1")))
	)
	)
)
#submitButton("submit", "Run prediction")

# Define server logic required to draw a histogram ----
server <- function(input, output, session) {

#Make sure nothing runs until the action button is pressed
observeEvent(
eventExpr=input$submit,
handlerExpr={

	#Get input data
	sData = c(input$n, input$t)

	#Run model
	dir_line = paste("python", 
		"dynamic_prediction_model/dynamic_prediction.py -n", 
		sData[1], 
		"-t", 
		sData[2])

	withProgress(message="Running XGBoost model", value=0, {
		system("echo source activate /home/dcusworth/.conda/envs/py36 > run.sh")
		system(paste("echo",dir_line,">> run.sh"))
		system("echo source deactivate >> run.sh")
		system("bash run.sh")
		incProgress(1, detail="done")
		}
	)

	#Get results
	dat = read.table("dynamic_prediction_model/results/individual_predictions.csv", 
		sep=",", header=T)

	output$plot1 <- renderPlot({
		#Plot results
		par(mar = c(5.1, 4.1, 3, 1))
		hist(dat$pred, breaks=0:12, xlim=c(0,12), xaxt="n", xlab="CinemaScore",
			main="Distribution of CinemaScore predictions", 
			col=adjustcolor("dodgerblue",.5), border="white")
		axis(1, at=0:12, 
			label=c("A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"))
		abline(v=mean(dat$pred,na.rm=T), lty=3, lwd=5, col="dodgerblue")
		abline(v=mean(dat$baseline,na.rm=T), lty=3, lwd=5, col="darkgreen")
		abline(v=quantile(dat$pred,.05,na.rm=T), lty=3, lwd=5, col="indianred2")
		abline(v=quantile(dat$pred,.95,na.rm=T), lty=3, lwd=5, col="indianred2")
		legend("topright", c("Mean", "Baseline", "5-95 percentile"), col=c("dodgerblue", 
			"darkgreen", "indianred2"), lwd=3, lty=2, bty="n", cex=1.5)
	})
	
	#Print text results
	text_results = readLines("dynamic_prediction_model/results/Prediction_Results.txt") 
	output$text1 = renderText({text_results})

}
)
}


shinyApp(ui = ui, server = server)



