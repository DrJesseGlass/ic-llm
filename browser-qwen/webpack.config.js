// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = (env, argv) => {
  const isDevelopment = argv.mode === 'development';

  return {
    mode: argv.mode || 'production',
    entry: './src/frontend/src/index.jsx',
    output: {
      path: path.resolve(__dirname, 'src/frontend/dist'),
      filename: 'bundle.[contenthash].js',
      clean: true,
    },
    resolve: {
      extensions: ['.js', '.jsx'],
    },
    module: {
      rules: [
        {
          test: /\.(js|jsx)$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: [
                '@babel/preset-env',
                ['@babel/preset-react', { runtime: 'automatic' }]
              ]
            }
          }
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader']
        },
        {
          test: /\.(wasm|gguf)$/,
          type: 'asset/resource',
          generator: {
            filename: 'assets/wasm/[name][ext]'
          }
        }
      ]
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './src/frontend/src/index.html',
        inject: 'body'
      }),
      new CopyWebpackPlugin({
        patterns: [
          {
            from: 'src/frontend/assets',
            to: 'assets'
          },
          {
            from: '.ic-assets.json',
            to: '.'
          }
        ]
      })
    ],
    devServer: {
      static: {
        directory: path.join(__dirname, 'src/frontend/dist'),
      },
      compress: true,
      port: 3000,
      hot: true,
      headers: {
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Opener-Policy': 'same-origin',
      }
    },
    experiments: {
      asyncWebAssembly: true,
    },
    performance: {
      hints: false,
      maxAssetSize: 700000000, // 700MB for the model file
      maxEntrypointSize: 700000000
    }
  };
};