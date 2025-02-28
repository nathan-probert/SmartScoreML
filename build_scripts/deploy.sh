#!/bin/bash

# one of {dev, prod}
ENV=${ENV:-dev}  # If ENV is not set, default to "dev"

MAX_ZIP_SIZE_MB=25

SOURCE_DIR="smartscore_ml"
OUTPUT_DIR="output"

STACK_NAME="SmartScore-ML-$ENV"
TEMPLATE_FILE="./template.yaml"

KEY="$STACK_NAME.zip"

LAMBDA_FUNCTIONS=(
  "MakePredictionsML-$ENV"
)


generate_smartscore_ml_stack() {
  if aws cloudformation describe-stacks --stack-name "$STACK_NAME" &>/dev/null; then
    echo "Updating CloudFormation stack $STACK_NAME..."
    UPDATE_OUTPUT=$(aws cloudformation update-stack \
      --stack-name "$STACK_NAME" \
      --template-body file://"$TEMPLATE_FILE" \
      --parameters ParameterKey=ENV,ParameterValue="$ENV" \
      --capabilities CAPABILITY_NAMED_IAM 2>&1)

    if echo "$UPDATE_OUTPUT" | grep -q "No updates are to be performed."; then
      echo "No updates needed. Skipping wait."
    else
      echo "Waiting for CloudFormation stack update to complete..."
      aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME"
    fi
  else
    echo "Creating CloudFormation stack $STACK_NAME..."
    aws cloudformation create-stack \
      --stack-name "$STACK_NAME" \
      --template-body file://"$TEMPLATE_FILE" \
      --parameters ParameterKey=ENV,ParameterValue="$ENV" \
      --capabilities CAPABILITY_NAMED_IAM

    echo "Waiting for CloudFormation stack creation to complete..."
    aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME"
  fi

  # Check the final status of the stack
  STACK_STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query "Stacks[0].StackStatus" --output text)

  if [[ "$STACK_STATUS" != "CREATE_COMPLETE" && "$STACK_STATUS" != "UPDATE_COMPLETE" ]]; then
    echo "CloudFormation stack operation failed with status: $STACK_STATUS."
    exit 1  # Exit with error
  fi

  echo "CloudFormation stack $STACK_NAME completed successfully with status: $STACK_STATUS."
}


generate_zip_file() {
  echo "Creating ZIP package for Lambda..."
  cd $OUTPUT_DIR

  # Exclude any .zip files from the ZIP package
  zip -r $KEY . -x "*.zip" > /dev/null

  cd ..

  ZIP_FILE_SIZE=$(stat -c%s "$OUTPUT_DIR/$KEY")
  ZIP_FILE_SIZE_MB=$((ZIP_FILE_SIZE / 1024 / 1024))

  echo "Size of ZIP file: $ZIP_FILE_SIZE_MB MB"

  if [ $ZIP_FILE_SIZE_MB -gt $MAX_ZIP_SIZE_MB ]; then
      echo "Error: The ZIP file exceeds $MAX_ZIP_SIZE_MB MB. Aborting deployment."
      exit 1
  fi
}


update_lambda_code() {
  for FUNCTION in "${LAMBDA_FUNCTIONS[@]}"; do
    echo "Updating Lambda function code: $FUNCTION..."

    aws lambda update-function-code \
      --function-name "$FUNCTION" \
      --zip-file fileb://$OUTPUT_DIR/$KEY &>/dev/null  # Suppress all output

    if [ $? -ne 0 ]; then
      echo "Error: Failed to update Lambda function code: $FUNCTION."
      exit 1
    fi

    echo "Lambda function code updated successfully: $FUNCTION."
  done
}


# main

# create the output directory
mkdir -p $OUTPUT_DIR

# update dependencies
poetry export -f requirements.txt --output $OUTPUT_DIR/requirements.txt --without-hashes --without dev
poetry run pip install --no-deps -r $OUTPUT_DIR/requirements.txt -t $OUTPUT_DIR
rm -f $OUTPUT_DIR/requirements.txt

# update the code
cp -r $SOURCE_DIR/* $OUTPUT_DIR/

# generate the ZIP file
generate_zip_file

# create the CloudFormation stack for smartscore_ml
generate_smartscore_ml_stack

# update the Lambda function code
update_lambda_code
