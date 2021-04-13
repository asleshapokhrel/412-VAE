{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "CSC412-PreProcessing.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPx0rDb9pDzDgrRPtCcHDoC"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XL_d4vNgCYrh",
        "outputId": "b26884ae-7aae-432a-eac6-b9bd9427f721"
      },
      "source": [
        "from google.colab import drive\n",
        "import requests\n",
        "import numpy as np\n",
        "import cv2 \n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "\n",
        "drive_name = '/content/drive'\n",
        "drive.mount(drive_name)\n",
        "\n",
        "drive_location = drive_name + '/My Drive/University of Toronto/4th Year/CSC412'\n",
        "data_location = drive_location + '/data/'"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rudR8GkX_QuQ"
      },
      "source": [
        "def download_img(object_name):\n",
        "    \"\"\"\n",
        "    Download .npy file for given object_name.\n",
        "    object_name (str) : Name of object drawing to be downoalded. \n",
        "                        E.g., \"airplane\", \"mouse\"\n",
        "    \"\"\"\n",
        "    url = \"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy\".format(object_name)\n",
        "    myfile = requests.get(url)\n",
        "    savename = data_location + '{}.npy'.format(object_name)\n",
        "\n",
        "    if os.path.exists(data_location + '{}.npy'.format(object_name)):\n",
        "        print(\"{}.npy file exists already\".format(object_name))\n",
        "    else:\n",
        "        with open(savename, 'wb') as f:\n",
        "            f.write(myfile.content)\n",
        "            print(\"{}.npy file download completed\".format(object_name))\n",
        "            f.close()\n",
        "\n",
        "\n",
        "def load_data(object_name):\n",
        "    \"\"\"\n",
        "    Return Nx784 numpy array corresponding to N drawings of size 28x28.\n",
        "    object_name (str) : Name of object drawing to be loaded to the memory. \n",
        "                        E.g., \"airplane\", \"mouse\"\n",
        "    \"\"\"\n",
        "    savename = data_location + '{}.npy'.format(object_name)\n",
        "    data = np.load(savename)\n",
        "\n",
        "    return data\n",
        "\n",
        "\n",
        "def display_img(im_arr):\n",
        "    \"\"\"\n",
        "    Plot im_arr.\n",
        "    im_arr (numpy uint8 array): Numpy array of size (784,) to be visualized.\n",
        "    \"\"\"\n",
        "    plt.imshow(im_arr.reshape(28,28), cmap='gray')"
      ],
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gnZ_6Wxz_y7y",
        "outputId": "4409ad16-2bb8-4659-caf0-c259995012b3"
      },
      "source": [
        "download_img('mouse')"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "mouse.npy file download completed\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "s2eXo11AHazW"
      },
      "source": [
        "data = load_data('airplane')"
      ],
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "id": "iO3n709IC5pG",
        "outputId": "e86b4b99-78b8-49cc-a7e2-083953ef0e71"
      },
      "source": [
        "display_img(data[1])"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}