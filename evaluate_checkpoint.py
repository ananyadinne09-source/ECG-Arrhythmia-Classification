from train_inception import load_data, build_model, test_model

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights, num_classes = load_data()
    model, criterion, optimizer, scheduler, scaler = build_model(num_classes, class_weights)
    test_model(model, test_loader)