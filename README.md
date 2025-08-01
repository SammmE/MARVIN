# OSCAR

**O**bservational **S**ystem for **C**onfiguring & **A**nalyzing **R**eal‑time nets

A powerful, interactive neural network training and visualization platform built with React, TensorFlow.js, and modern web technologies. OSCAR provides real-time insights into neural network behavior, making machine learning education and experimentation more accessible and intuitive.

## ✨ Features

### 🎯 Interactive Neural Network Training

- **Real-time Training**: Watch your neural networks learn with live metric updates
- **Visual Feedback**: See weights, activations, and predictions update in real-time
- **Training Controls**: Start, pause, resume, and step through training epochs
- **Multiple Problem Types**: Support for both regression and classification tasks

### 📊 Advanced Visualization

- **Network Architecture**: Interactive visualization of neural network layers and connections
- **Layer Activations**: Real-time display of neuron activations across all layers
- **Weight Visualization**: Live weight updates with visual thickness representing magnitude
- **Training Metrics**: Comprehensive charts for loss, accuracy, and validation metrics
- **Prediction Overlay**: See model predictions overlaid on your data

### 🔧 Flexible Configuration

- **Hyperparameter Tuning**: Easy-to-use interface for adjusting learning rates, optimizers, and architectures
- **Custom Architectures**: Build networks with Dense, Conv1D, Conv2D, and Flatten layers
- **Data Management**: Import your own datasets or use built-in sample data
- **Activation Functions**: Support for ReLU, Sigmoid, Tanh, Softmax, and more

### 🚀 Performance Optimized

- **Web Workers**: Training runs in background workers to keep UI responsive
- **TensorFlow.js**: Leverages GPU acceleration when available
- **Efficient Rendering**: Optimized visualization updates for smooth performance

## 🛠️ Tech Stack

- **Frontend**: React 19, TypeScript, Tailwind CSS
- **Machine Learning**: TensorFlow.js, TensorFlow.js Visualization
- **Visualization**: Recharts, React Flow, HTML2Canvas
- **State Management**: Zustand with LocalForage persistence
- **UI Components**: Radix UI primitives with custom styling
- **Build Tool**: Vite with SWC compilation
- **Code Quality**: Biome for linting and formatting

## 📦 Installation

### Prerequisites

- Node.js 18+
- npm, yarn, or bun

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/SammmE/MARVIN.git
   cd MARVIN
   ```

2. **Install dependencies**

   ```bash
   npm install
   # or
   yarn install
   # or
   bun install
   ```

3. **Start the development server**

   ```bash
   npm run dev
   # or
   yarn dev
   # or
   bun dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:5173` to start using OSCAR

## 🚀 Quick Start

### Basic Workflow

1. **Data Tab**: Import your dataset or use sample data (sinusoid, spiral, etc.)
2. **Hyperparameters Tab**: Configure your network architecture and training parameters
3. **Model Tab**: Visualize your network architecture and see layer details
4. **Training Tab**: Train your model with real-time monitoring and controls

### Example: Training a Simple Regression Model

1. Go to the **Data** tab and select the "Sinusoid" preset
2. Switch to **Hyperparameters** and set:
   - 2 Dense layers with 64 and 32 units
   - ReLU activation functions
   - Learning rate: 0.001
   - Adam optimizer
3. Navigate to **Training** and click "Start Training"
4. Watch as your model learns to approximate the sine function!

## 📁 Project Structure

```text
src/
├── components/          # Reusable UI components
│   ├── ui/             # Base UI components (buttons, cards, etc.)
│   ├── activation-*.tsx # Activation function visualizations
│   ├── data-*.tsx      # Data management components
│   ├── network-*.tsx   # Network visualization components
│   └── training-*.tsx  # Training control components
├── lib/                # Core logic and state management
│   ├── oscar-store.ts  # 🎯 UNIFIED STORE - All application state management
│   ├── data-store.ts   # ⚠️ DEPRECATED - Use oscar-store.ts instead
│   ├── hyperparameters-store.ts # ⚠️ DEPRECATED - Use oscar-store.ts instead
│   ├── model-store.ts  # ⚠️ DEPRECATED - Use oscar-store.ts instead
│   ├── training-store.ts # ⚠️ DEPRECATED - Use oscar-store.ts instead
│   └── tensorflow-monitor.ts # TensorFlow.js integration
├── pages/              # Main application pages
│   ├── data.tsx        # Data import and preview
│   ├── hyperparameters.tsx # Network configuration
│   ├── model.tsx       # Model visualization
│   └── training.tsx    # Training interface
└── workers/            # Web workers for background processing
    └── training-worker.ts # Neural network training worker
```

## 🎛️ Configuration

### Supported Layer Types

- **Dense**: Fully connected layers with configurable units and activation
- **Conv1D**: 1D convolutional layers for sequence data
- **Conv2D**: 2D convolutional layers for image data
- **Flatten**: Flatten multi-dimensional input to 1D

### Activation Functions

- ReLU, Leaky ReLU, ELU
- Sigmoid, Tanh
- Softmax (for classification)
- Linear (for regression output)

### Optimizers

- Adam (recommended for most cases)
- SGD with momentum
- RMSprop
- Adagrad

## 🔍 Advanced Features

### Real-time Monitoring

- **Live Metrics**: Loss and accuracy updates every epoch
- **Activation Heatmaps**: Visualize neuron activations across layers
- **Weight Tracking**: Monitor weight changes during training
- **Prediction Visualization**: See model predictions on your data

### Training Controls

- **Speed Control**: Adjust training speed from real-time to manual stepping
- **Epoch Scrubbing**: Jump to any point in training history
- **Batch Stepping**: Step through training batch by batch
- **Error Handling**: Automatic detection and fixes for common issues

### Data Management

- **Multiple Formats**: Support for CSV and JSON data import
- **Feature Selection**: Choose input features and target variables
- **Data Preprocessing**: Automatic normalization and validation
- **Sample Datasets**: Built-in datasets for quick experimentation

## 🐛 Troubleshooting

### Common Issues

#### Training won't start

- Check that you have data loaded in the Data tab
- Ensure your network architecture is valid in Hyperparameters
- Verify that input/output dimensions match your data

#### Performance issues

- Reduce network size for complex architectures
- Lower the training speed setting
- Close other browser tabs to free up resources

#### Browser compatibility

- Modern browsers with WebGL support required
- Chrome/Edge recommended for best performance
- Firefox and Safari supported with possible performance limitations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and formatting (using Biome)
- Pull request process
- Issue reporting
- Feature requests

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run linting: `npm run lint`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow.js** team for the amazing ML framework
- **React** and **Vite** communities for excellent development tools
- **Radix UI** for accessible component primitives
- **Recharts** for beautiful data visualization components

## 📚 Learn More

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Neural Network Fundamentals](https://www.3blue1brown.com/neural-networks)
- [React + TypeScript Best Practices](https://react-typescript-cheatsheet.netlify.app/)

---

Made with ❤️ for machine learning education and experimentation
