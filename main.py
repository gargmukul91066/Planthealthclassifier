from utils.data_loader import load_data
from models.train_models import train_and_evaluate

if __name__ == "__main__":
    csv_path = "data/train.csv"
    image_dir = "data/images"
    
    print("Loading data and extracting features...")
    X, y, le = load_data(csv_path, image_dir)

    print("Training models...")
    train_and_evaluate(X, y, le)
