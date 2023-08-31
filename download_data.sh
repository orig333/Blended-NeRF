dname=$1

case $dname in
    "nerf_synthetic")
        mkdir -p data
        mkdir -p data/blender
        gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
        unzip nerf_synthetic.zip
        rm -rf __MACOSX
        mv nerf_synthetic data/blender
        rm nerf_synthetic.zip
        ;;
    "nerf_llff")
        mkdir -p data
        mkdir -p data/llff
        gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
        unzip nerf_llff_data.zip
        rm -rf __MACOSX
        mv nerf_llff_data data/llff
        rm nerf_llff_data.zip
        ;;
    "nerf_real_360")
        mkdir -p data
        mkdir -p data/nerf_360
        gdown 1jzggQ7IPaJJTKx9yLASWHrX8dXHnG5eB
        unzip nerf_real_360.zip
        rm -rf __MACOSX
        mkdir nerf_real_360
        mv vasedeck nerf_real_360
        mv pinecone nerf_real_360
        mv nerf_real_360 data/nerf_360
        rm nerf_real_360.zip
        ;;
esac