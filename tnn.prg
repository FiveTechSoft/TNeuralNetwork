#include "hbclass.ch"

function Main()

   local oNeuralNetwork, n 

   SET DECIMALS TO 9

   oNeuralNetwork = TNeuralNetwork():New( 2, 2, 2 )

   for n = 1 to 50
      oNeuralNetwork:Train( { 0.05, 0.1 }, { 0.01, 0.99 } )
      ? oNeuralNetwork:CalculateTotalError( { { 0.05, 0.1 }, { 0.01, 0.99 } } )
   next   

return nil   

CLASS TNeuralNetwork

   DATA   nLearningRate INIT 0.5

   DATA   nInputs

   DATA   oHiddenLayer

   DATA   oOutputLayer

   METHOD New( nInputs, nHidden, nOutputs, aHiddenLayerWeights, aHiddenLayerBias, aOutputLayerWeights, aOutputLayerBias )

   METHOD init_weights_from_inputs_to_hidden_layer_neurons( aHiddenLayerWeights )

   METHOD init_weights_from_hidden_layer_neurons_to_output_layer_neurons( aOutputLayerWeights )

   METHOD Inspect()

   METHOD FeedForward( nInputs )

   METHOD Train( aTrainingInputs, aTrainingOutputs )   

   METHOD CalculateTotalError( aTrainingSets )

ENDCLASS

METHOD New( nInputs, nHidden, nOutputs, aHiddenLayerWeights, aHiddenLayerBias, aOutputLayerWeights, aOutputLayerBias ) ;
   CLASS TNeuralNetwork

   ::nInputs = nInputs
   ::oHiddenLayer = TNeuronLayer():New( nHidden, aHiddenLayerBias )
   ::oOutputLayer = TNeuronLayer():New( nOutputs, aOutputLayerBias )
   ::init_weights_from_inputs_to_hidden_layer_neurons( aHiddenLayerWeights )
   ::init_weights_from_hidden_layer_neurons_to_output_layer_neurons( aOutputLayerWeights )

return Self

METHOD init_weights_from_inputs_to_hidden_layer_neurons( aHiddenLayerWeights ) CLASS TNeuralNetwork

   local nWeight := 1, n, m

   for n = 1 to Len( ::oHiddenLayer:aNeurons )
      for m = 1 to ::nInputs
        if Empty( aHiddenLayerWeights )
           AAdd( ::oHiddenLayer:aNeurons[ n ]:aWeights, hb_random() )
        else
           AAdd( ::oHiddenLayer:aNeurons[ n ]:aWeights, aHiddenLayerWeights[ nWeigth ] )
        endif   
      next
      nWeight++
   next
   
return nil   

METHOD init_weights_from_hidden_layer_neurons_to_output_layer_neurons( aOutputLayerWeights ) CLASS TNeuralNetwork

   local nWeight := 1, n, m

   for n = 1 to Len( ::oOutputLayer:aNeurons )
      for m = 1 to Len( ::oHiddenLayer:aNeurons )
        if Empty( aOutputLayerWeights )
           AAdd( ::oOutputLayer:aNeurons[ n ]:aWeights, hb_random() )
        else
           AAdd( ::oOutputLayer:aNeurons[ n ]:aWeights, aOutputLayerWeights[ nWeigth ] )
        endif   
      next
      nWeight++
   next
   
return nil   

METHOD Inspect() CLASS TNeuralNetwork

   local cInfo := "Inputs: " + AllTrim( Str( ::nInputs ) ) + hb_OsNewLine() + "======" + hb_OsNewLine()

   cInfo += "Hidden Layer: " + hb_OsNewLine() + ::oHiddenLayer:Inspect() + "==============" + hb_OsNewLine()
   cInfo += "Output Layer: " + hb_OsNewLine() + ::oOutputLayer:Inspect() + "==============" + hb_OsNewLine()                    

   ? cInfo

return nil   

METHOD FeedForward( nInputs ) CLASS TNeuralNetwork 

   local aHiddenLayerOutputs := ::oHiddenLayer:FeedForward( nInputs )

return ::oOutputLayer:FeedForward( aHiddenLayerOutputs )

METHOD Train( aTrainingInputs, aTrainingOutputs ) CLASS TNeuralNetwork 

   local aPd_errors_wrt_output_neuron_total_net_input := AFill( Array( Len( ::oOutputLayer:aNeurons ) ), 0 )
   local aPd_errors_wrt_hidden_neuron_total_net_input := AFill( Array( Len( ::oHiddenLayer:aNeurons ) ), 0 )
   local n, m, d_error_wrt_hidden_neuron_output, pd_error_wrt_weight

   ::FeedForward( aTrainingInputs )

   for n = 1 to Len( ::oOutputLayer:aNeurons )
      aPd_errors_wrt_output_neuron_total_net_input[ n ] = ::oOutputLayer:aNeurons[ n ]:CalculatePdErrorWrtTotalNetInput( aTrainingOutputs[ n ] )
   next

   for n = 1 to Len( ::oHiddenLayer:aNeurons )
      d_error_wrt_hidden_neuron_output = 0
      for m = 1 to Len( ::oOutputLayer:aNeurons )
         d_error_wrt_hidden_neuron_output += aPd_errors_wrt_output_neuron_total_net_input[ m ] * ::oOutputLayer:aNeurons[ m ]:aWeights[ n ]
      next

      aPd_errors_wrt_hidden_neuron_total_net_input[ n ] = d_error_wrt_hidden_neuron_output * ::oHiddenLayer:aNeurons[ n ]:CalculatePdTotalNetInputWrtInput()
   next

   for n = 1 to Len( ::oOutputLayer:aNeurons )
      for m = 1 to Len( ::oOutputLayer:aNeurons[ n ]:aweights )
         pd_error_wrt_weight = aPd_errors_wrt_output_neuron_total_net_input[ n ] * ::oOutputLayer:aNeurons[ n ]:CalculatePdTotalNetInputWrtWeight( m )
         ::oOutputLayer:aNeurons[ n ]:aWeights[ m ] -= ::nLearningRate * pd_error_wrt_weight
      next
   next      

   for n = 1 to Len( ::oHiddenLayer:aNeurons )
      for m = 1 to Len( ::oHiddenLayer:aNeurons[ n]:aWeights )
         pd_error_wrt_weight = aPd_errors_wrt_hidden_neuron_total_net_input[ n ] * ::oHiddenLayer:aNeurons[ n ]:CalculatePdTotalNetInputWrtWeight( m )
         ::oHiddenLayer:aNeurons[ n ]:aWeights[ m ] -= ::nLearningRate * pd_error_wrt_weight
      next   
   next

return nil   

METHOD CalculateTotalError( aTrainingSets ) CLASS TNeuralNetwork

   local total_error := 0
   local training_inputs, training_outputs

   for n = 1 to Len( aTrainingSets )
      training_inputs :=  training_outputs := aTrainingSets[ n ]
      ::FeedForward( training_inputs )
      for m = 1 to Len( training_outputs )
         total_error += ::oOutputLayer:aNeurons[ m ]:CalculateError( training_outputs[ m ] )
      next
   next        

return total_error

CLASS TNeuronLayer

   DATA  nBias
   DATA  aNeurons INIT {}

   METHOD New( nNeurons, nBias )

   METHOD Inspect()

   METHOD FeedForward( aInputs )

   METHOD GetOutputs()

ENDCLASS   

METHOD New( nNeurons, nBias ) CLASS TNeuronLayer

   local n

   ::nBias = If( ! Empty( nBias ), nBias, hb_Random() )

   for n = 1 to nNeurons
      AAdd( ::aNeurons, TNeuron():New( ::nBias ) )
   next   

return Self 

METHOD Inspect() CLASS TNeuronLayer

   local cInfo := "Neurons: " + AllTrim( Str( Len( ::aNeurons ) ) ) + hb_OsNewLine()
   local oNeuron, nWeight

   for each oNeuron in ::aNeurons
      cInfo += "   neuron: " + AllTrim( Str( oNeuron:__enumIndex ) ) + hb_OsNewLine()
      for each nWeight in oNeuron:aWeights
         cInfo += "      weight: " + AllTrim( Str( nWeight ) ) + hb_OsNewLine()
      next
      cInfo += "      Bias: " + AllTrim( Str( ::nBias ) ) + hb_OsNewLine()
   next      

return cInfo  

METHOD FeedForward( aInputs ) CLASS TNeuronLayer

   local oNeuron, aOutputs := {}

   for each oNeuron in ::aNeurons
      AAdd( aOutputs, oNeuron:CalculateOutput( aInputs ) )
   next 

return aOutputs

METHOD GetOutputs() CLASS TNeuronLayer 

   local oNeuron, aOutputs := {}

   for each oNeuron in ::aNeurons
      AAdd( aOutputs, oNeuron:nOutput )
   next 

return aOutputs

CLASS TNeuron

   DATA   nBias
   DATA   aWeights INIT {}
   DATA   aInputs
   DATA   nOutput

   METHOD New( nBias )

   METHOD CalculateOutput( nInputs )

   METHOD CalculateTotalNetInput()

   METHOD Squash( nTotalNetInput )

   METHOD CalculatePdErrorWrtTotalNetInput( nTargetOutput )

   METHOD CalculateError( nTargetOutput )

   METHOD CalculatePdErrorWrtOutput( nTargetOutput )

   METHOD CalculatePdTotalNetInputWrtInput()

   METHOD CalculatePdTotalNetInputWrtWeight( nIndex )

ENDCLASS

METHOD New( nBias ) CLASS TNeuron 

   ::nBias = nBias

return Self

METHOD CalculateOutput( aInputs ) CLASS TNeuron 

   ::aInputs = aInputs
   ::nOutput = ::Squash( ::CalculateTotalNetInput() )

return ::nOutput

METHOD CalculateTotalNetInput() CLASS TNeuron 

   local nTotal := 0, n

   for n = 1 to Len( ::aInputs )
      nTotal += ::aInputs[ n ] * ::aWeights[ n ]
   next 

return nTotal + ::nBias

METHOD Squash( nTotalNetInput ) CLASS TNeuron 

return 1 / ( 1 + Math_E() ^ -nTotalNetInput )

METHOD CalculatePdErrorWrtTotalNetInput( nTargetOutput ) CLASS TNeuron

return ::CalculatePdErrorWrtOutput( nTargetOutput ) * ::CalculatePdTotalNetInputWrtInput()

METHOD CalculateError( nTargetOutput ) CLASS TNeuron 

return 0.5 * ( nTargetOutput - ::nOutput ) ^ 2

METHOD CalculatePdErrorWrtOutput( nTargetOutput ) CLASS TNeuron 

return -( nTargetOutput - ::nOutput )

METHOD CalculatePdTotalNetInputWrtInput() CLASS TNeuron  

return ::nOutput * ( 1 - ::nOutput )

METHOD CalculatePdTotalNetInputWrtWeight( nIndex ) CLASS TNeuron 

return ::aInputs[ nIndex ]

#pragma BEGINDUMP

#include <hbapi.h>
#include <math.h>

#ifndef M_E 
   #define M_E  2.71828182845904523536
#endif   

HB_FUNC( MATH_E )
{
   hb_retnd( M_E );
}

#pragma ENDDUMP