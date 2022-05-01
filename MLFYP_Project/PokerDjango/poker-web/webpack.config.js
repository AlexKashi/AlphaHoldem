const path = require('path')

module.exports = {

  entry: path.resolve(__dirname, 'src', 'index.js'),
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js'
  },
  resolve: {
    extensions: ['', '.js', '.jsx']
  },

  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.js|\.jsx$/,
        include: path.resolve(__dirname, 'src'),
        exclude: /node_modules/,
        use: [{
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/react']
            ]
          }
        }]
      }
    ]

  }
  // module: {
  //   rules: [
  //     {
  //       test: /\.js|\.jsx$/,
  //       loader: 'babel-loader',
  //       exclude: /node_modules/,
  //       options: {
  //         presets: ['react', 'es2015']
  //       }
  //     }
  //   ]
  // }

  
}