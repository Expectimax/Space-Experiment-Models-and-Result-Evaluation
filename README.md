Hier sind für dich eigentlich nur 8 Files besonders interessant:

Die Files nach dem Schema "Model_formal.py" sind dazu da die Models neu zu trainieren. Falls du das machen willst brauchst du natürlich erstmal neue Bilder. 
Ist ein bisschen tricky die Datasets dann in der richtigen Struktur zu haben, aber das kriegt man hin mit ein bisschen googeln. 
Anschließend einfach laufen lassen mit einer hohen Anzahl an Epochen und checken ab wie vielen Epochen die Validation Accuracy steigt, aber die Test Accuracy fällt.
Anschließend diese Anzahl an Epochen festlegen und nochmals durchlaufen lassen und das Model abspeichern.
Ich habe keinen automatisierten Stopp porgammiert. Ich hab ziemlich viel Drop out in den Models und das führt dazu dass die Abbruchbedingung viel zu früh zufällig erreicht wird.

Die Files nach dem Schema "Formal_Model.kers" sind die fertig trainierten Models. Die kannst du einfach in die file "predict_test_set" hochladen. 
Dann das Test Set definieren und mit den Models laufen lassen. Dann klassifizieren die Models die Klasse der Bilder im Test Set.
