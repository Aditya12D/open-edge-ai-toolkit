import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class ModelConfig {
  final String modelPath;
  final List<String> classes;

  ModelConfig({required this.modelPath, required this.classes});
}

class ModelService {
  Interpreter? _interpreter;
  List<String> _currentClasses = [];

  final Map<String, ModelConfig> models = {
    'Tomato': ModelConfig(
      modelPath: 'assets/tomato.tflite',
      classes: [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted",
        "Tomato_Target_Spot",
        "Tomato_Tomato_YellowLeaf_Curl_Virus",
        "Tomato_Tomato_mosaic_virus",
        "Tomato_healthy"
      ]
    ),
    'Wheat': ModelConfig(
      modelPath: 'assets/wheat.tflite',
      classes: [
        "healthy",
        "mildew",
        "yellow_rust",
        "brown_rust"
      ]
    ),
  };

  void dispose() {
    _interpreter?.close();
  }

  Future<void> loadModel(String crop) async {
    final config = models[crop];
    if (config == null) throw Exception("Model config not found for $crop");

    // Close existing interpreter if open
    _interpreter?.close();

    _interpreter = await Interpreter.fromAsset(config.modelPath);
    _currentClasses = config.classes;
  }

  String formatLabel(String label) {
    if (label.startsWith("Tomato_")) {
      label = label.substring(7);
    }
    label = label.replaceAll("_", " ");
    return label;
  }

  Future<Map<String, dynamic>> predict(File image) async {
    if (_interpreter == null || _currentClasses.isEmpty) {
      throw Exception('Model not loaded');
    }

    final decodedImage = img.decodeImage(await image.readAsBytes());
    if (decodedImage == null) throw Exception('Failed to decode image');

    final resizedImage = img.copyResize(decodedImage, width: 224, height: 224);

    final inputTensor = _interpreter!.getInputTensor(0);
    final isFloat = inputTensor.type == TfLiteType.kTfLiteFloat32 ||
        inputTensor.type == TfLiteType.kTfLiteFloat16;

    var input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resizedImage.getPixel(x, y);
            final r = pixel.r;
            final g = pixel.g;
            final b = pixel.b;
            
            if (isFloat) {
              return [r.toDouble(), g.toDouble(), b.toDouble()];
            } else {
              return [r.toInt(), g.toInt(), b.toInt()];
            }
          },
        ),
      ),
    );

    int numClasses = _currentClasses.length;
    var output = List.generate(1, (_) => List.filled(numClasses, 0.0));

    _interpreter!.run(input, output);

    final probabilities = output[0];
    int maxIndex = 0;
    double maxProb = probabilities[0];
    
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    return {
      'label': formatLabel(_currentClasses[maxIndex]),
      'confidence': maxProb
    };
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Disease Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const DiseaseDetectorScreen(),
    );
  }
}

class DiseaseDetectorScreen extends StatefulWidget {
  const DiseaseDetectorScreen({super.key});

  @override
  State<DiseaseDetectorScreen> createState() => _DiseaseDetectorScreenState();
}

class _DiseaseDetectorScreenState extends State<DiseaseDetectorScreen> {
  final ModelService _modelService = ModelService();
  final ImagePicker _picker = ImagePicker();
  
  String _selectedCrop = 'Tomato';
  final List<String> _crops = ['Tomato', 'Wheat'];

  File? _image;
  String? _prediction;
  double? _confidence;
  bool _isProcessing = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadSelectedModel();
  }

  Future<void> _loadSelectedModel() async {
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });
    try {
      await _modelService.loadModel(_selectedCrop);
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load model: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _prediction = null;
          _confidence = null;
        });
      }
    }
  }

  void _onCropChanged(String? newCrop) {
    if (newCrop != null && newCrop != _selectedCrop) {
      setState(() {
        _selectedCrop = newCrop;
      });
      _loadSelectedModel();
    }
  }

  Future<void> _pickImage() async {
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      if (!mounted) return;
      setState(() {
        _image = File(pickedFile.path);
        _prediction = null;
        _confidence = null;
        _errorMessage = null;
        _isProcessing = true;
      });
      _runInference(_image!);
    }
  }

  Future<void> _runInference(File image) async {
    try {
      final result = await _modelService.predict(image);
      setState(() {
        _prediction = result['label'];
        _confidence = result['confidence'];
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Error processing image: $e';
        _isProcessing = false;
      });
      debugPrint('Inference error: $e');
    }
  }

  @override
  void dispose() {
    _modelService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Disease Detector'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 8.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Select Crop: ',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(width: 10),
                    DropdownButton<String>(
                      value: _selectedCrop,
                      items: _crops.map((String value) {
                        return DropdownMenuItem<String>(
                          value: value,
                          child: Text(value, style: const TextStyle(fontSize: 18)),
                        );
                      }).toList(),
                      onChanged: _isProcessing ? null : _onCropChanged,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              
              if (_errorMessage != null)
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.red, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                ),

              if (_image != null)
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Image.file(_image!, height: 300),
                )
              else
                const Icon(Icons.image, size: 100, color: Colors.grey),
                
              const SizedBox(height: 20),
              if (_isProcessing)
                const CircularProgressIndicator()
              else if (_prediction != null)
                Column(
                  children: [
                    Text(
                      'Prediction: $_prediction',
                      style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.green),
                      textAlign: TextAlign.center,
                    ),
                    if (_confidence != null)
                      Text(
                        'Confidence: ${(_confidence! * 100).toStringAsFixed(2)}%',
                        style: const TextStyle(fontSize: 16),
                      ),
                  ],
                ),
              const SizedBox(height: 30),
              ElevatedButton.icon(
                onPressed: _isProcessing ? null : _pickImage,
                icon: const Icon(Icons.photo_library),
                label: const Text('Select Leaf Image'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
