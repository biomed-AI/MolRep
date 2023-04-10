import concurrent.futures
import time
import os
import pickle

TEMP_DIRECTORY = "_temp"
TEMP_FILE_NAMES = "results_"

SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT = 5


def parallel_execution(function_to_call, arguments_dict, max_workers=1, maximum_time_per_function_call=10,
                       save_temporary_results=True, save_method=None, save_memory=False):
    # create directory for storing temp files
    if save_temporary_results:
        if not os.path.exists(TEMP_DIRECTORY):
            os.makedirs(TEMP_DIRECTORY)

    file_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        result_by_key = {}
        keys_with_error = []
        success_message_percentile = SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT / 100
        next_finished_message = success_message_percentile
        success_message_threshold = success_message_percentile
        number_of_function_calls_finished = 0

        futures_to_key = {executor.submit(function_to_call, *arguments_dict[key]): key for
                          key in arguments_dict}

        print("Started parallel execution at", time.ctime(), "\n", len(arguments_dict),
              "number of functions to call")
        try:
            for future in concurrent.futures.as_completed(futures_to_key,
                                                          timeout=maximum_time_per_function_call * len(
                                                              arguments_dict)):
                key = futures_to_key[future]

                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (key, exc))
                    keys_with_error.append(key)
                else:
                    result_by_key[key] = result

                del futures_to_key[future]

                if len(result_by_key) / len(arguments_dict) > success_message_threshold:
                    if save_temporary_results:
                        file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + str(
                            round(next_finished_message * 100))
                        if save_method is None:
                            try:
                                with open(file_path, "wb") as file:
                                    pickle.dump(result_by_key, file)
                            except pickle.PickleError:
                                print("Save not worked")
                            else:
                                file_paths.append(file_path)
                                print("Saved file:", file_path)
                        else:
                            file_path = save_method(result_by_key, file_path)
                            file_paths.append(file_path)
                            print("Saved file:", file_path)
                        if save_memory:
                            number_of_function_calls_finished += len(result_by_key)
                            # delete saved results
                            del result_by_key
                            result_by_key = {}
                            success_message_threshold = 0
                        else:
                            number_of_function_calls_finished = len(result_by_key)

                    success_message_threshold += success_message_percentile
                    print(round(number_of_function_calls_finished / len(arguments_dict) * 100), "% completed at",
                          time.ctime())
                    next_finished_message += success_message_percentile
        except concurrent.futures.TimeoutError:
            keys_with_error.extend(set(futures_to_key.values()).difference(result_by_key.keys()))
            print(str(len(keys_with_error)) + " function calls of" + str(
                function_to_call) + "did not finished in a total of "
                  + str(
                maximum_time_per_function_call * len(arguments_dict)) + "seconds. Not finished argument keys: "
                  + str(keys_with_error))
        except Exception as exc:
            print('Generated an exception: ', exc)

    return result_by_key, keys_with_error
