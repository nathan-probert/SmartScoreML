from smartscore_info_client.schemas.db_player_info import PlayerDbInfo


# This is meant to be ran locally, and then the model is to be uploaded to the cloud and accessed there
def create_model():
    # Load the data



    try:
        data = pd.read_csv(filename, encoding="latin1")
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(filename, encoding="utf-8")
        except UnicodeDecodeError:
            print("Manually save the csv file and try again.\n")
            exit(1)

    # Drop the rows where 'Scored' is empty
    statsToView = ['GPG', 'Last 5 GPG', 'OTGA']

    data = data[data['Scored'] != ' ']
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Preprocess the data
    features = data[statsToView]
    # features = data[['GPG','Last 5 GPG','HGPG','PPG','OTPM','TGPG','OTGA','Home (1)']]
    labels = data['Scored']

    # Split the data into training and testing sets
    testSize = 0.2
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=testSize, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the input shape
    input_shape = (X_train.shape[1],)
    inputs = Input(shape=input_shape)

    # Define the layers
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a specified learning rate
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)

    print(f'Test loss: {loss}')

    # Save the model
    model.save("randomModel")