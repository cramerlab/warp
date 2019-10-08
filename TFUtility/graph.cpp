#define _CRT_SECURE_NO_WARNINGS
#include "Functions.h"

using tensorflow::AllocationDescription;
using tensorflow::DataType;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::MetaGraphDef;
using tensorflow::NameRangeMap;
using tensorflow::NameRangesForNode;
using tensorflow::NewSession;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpRegistry;
using tensorflow::PartialTensorShape;
using tensorflow::RunMetadata;
using tensorflow::RunOptions;
using tensorflow::SavedModel;
using tensorflow::SavedModelBundle;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::TensorId;
using tensorflow::TensorShape;
using tensorflow::TensorShapeProto;
using tensorflow::VersionDef;
using tensorflow::error::Code;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;
using tensorflow::gtl::ArraySlice;
using tensorflow::mutex_lock;
using tensorflow::string;
using tensorflow::strings::StrCat;

/// SavedModel assets directory.
constexpr char kSavedModelAssetsDirectory[] = "assets";

/// SavedModel assets.extra directory.
constexpr char kSavedModelAssetsExtraDirectory[] = "assets.extra";

/// SavedModel assets key for graph collection-def.
constexpr char kSavedModelAssetsKey[] = "saved_model_assets";

/// SavedModel proto filename.
constexpr char kSavedModelFilenamePb[] = "saved_model.pb";

/// SavedModel text format proto filename.
constexpr char kSavedModelFilenamePbTxt[] = "saved_model.pbtxt";

/// SavedModel legacy init op key.
constexpr char kSavedModelLegacyInitOpKey[] = "legacy_init_op";

/// SavedModel main op key.
constexpr char kSavedModelMainOpKey[] = "saved_model_main_op";

/// Directory in which to save the SavedModel variables.
constexpr char kSavedModelVariablesDirectory[] = "variables";

/// SavedModel variables filename.
constexpr char kSavedModelVariablesFilename[] = "variables";

Status ReadSavedModel(const string& export_dir, SavedModel* saved_model_proto) 
{
    const string saved_model_pb_path = tensorflow::io::JoinPath(export_dir, kSavedModelFilenamePb);
    if (tensorflow::Env::Default()->FileExists(saved_model_pb_path).ok())
        return ReadBinaryProto(tensorflow::Env::Default(), saved_model_pb_path, saved_model_proto);

    const string saved_model_pbtxt_path = tensorflow::io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
    if (tensorflow::Env::Default()->FileExists(saved_model_pbtxt_path).ok())
        return ReadTextProto(tensorflow::Env::Default(), saved_model_pbtxt_path,
            saved_model_proto);

    return Status(Code::NOT_FOUND,
                  "Could not find SavedModel .pb or .pbtxt at supplied export "
                  "directory path: " +
                  export_dir);
}

Status FindMetaGraphDefToLoad(const SavedModel& saved_model_proto,
                              const std::unordered_set<string>& tags,
                              MetaGraphDef* meta_graph_def_to_load) 
{
  for (const MetaGraphDef& meta_graph_def : saved_model_proto.meta_graphs()) 
  {
    // Get tags from the meta_graph_def.
    std::unordered_set<string> graph_tags;
    for (const string& tag : meta_graph_def.meta_info_def().tags())
      graph_tags.insert(tag);

    // Match with the set of tags provided.
    if (graph_tags == tags) 
    {
      *meta_graph_def_to_load = meta_graph_def;
      return Status::OK();
    }
  }

  string tags_as_string = "{ ";
  for (const string& tag : tags)
    tags_as_string = tensorflow::strings::StrCat(tags_as_string, tag, " ");

  tags_as_string = tensorflow::strings::StrCat(tags_as_string, "}");
  return Status(Code::NOT_FOUND,
                "Could not find meta graph def matching supplied tags: " +
                tags_as_string +
                ". To inspect available tag-sets in the SavedModel, please "
                "use the SavedModel CLI: `saved_model_cli`");
}

Status LoadMetaGraphIntoSession(const MetaGraphDef& meta_graph_def,
                                const SessionOptions& session_options,
                                std::unique_ptr<Session>* session) 
{
  session->reset(NewSession(session_options));
  return (*session)->Create(meta_graph_def.graph_def());
}

Tensor CreateStringTensor(const string& value) 
{
  Tensor tensor(DataType::DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

void AddAssetsTensorsToInputs(const tensorflow::StringPiece export_dir,
                              const std::vector<tensorflow::AssetFileDef>& asset_file_defs,
                              std::vector<std::pair<string, Tensor>>* inputs) 
{
  if (asset_file_defs.empty())
    return;

  for (auto& asset_file_def : asset_file_defs) 
  {
    Tensor assets_file_path_tensor = CreateStringTensor(tensorflow::io::JoinPath(export_dir, kSavedModelAssetsDirectory, asset_file_def.filename()));
    inputs->push_back({ asset_file_def.tensor_info().name(), assets_file_path_tensor });
  }
}

bool HasMainOp(const MetaGraphDef& meta_graph_def) 
{
  const auto& collection_def_map = meta_graph_def.collection_def();
  if (collection_def_map.find(kSavedModelMainOpKey) !=
    collection_def_map.end()) {
    return true;
  }
  return false;
}

Status RunMainOp(const RunOptions& run_options, const string& export_dir,
                 const MetaGraphDef& meta_graph_def,
                 const std::vector<tensorflow::AssetFileDef>& asset_file_defs,
                 Session* session) 
{
  LOG(INFO) << "Running MainOp on SavedModel bundle.";
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto main_op_it = collection_def_map.find(kSavedModelMainOpKey);
  if (main_op_it != collection_def_map.end()) 
  {
    if (main_op_it->second.node_list().value_size() != 1)
      return FailedPrecondition(tensorflow::strings::StrCat("Expected exactly one main op in : ", export_dir));

    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    const tensorflow::StringPiece main_op_name = main_op_it->second.node_list().value(0);
    return session->Run(run_options, inputs, {}, { main_op_name.ToString() },
                        nullptr /* outputs */, &run_metadata);
  }
  return Status::OK();
}

Status RunRestore(const RunOptions& run_options, const string& export_dir,
                  const tensorflow::StringPiece restore_op_name,
                  const tensorflow::StringPiece variable_filename_const_op_name,
                  const std::vector<tensorflow::AssetFileDef>& asset_file_defs,
                  Session* session) 
{
  LOG(INFO) << "Restoring SavedModel bundle.";
  // Find path to variables to be restored in export directory.
  const string variables_directory = tensorflow::io::JoinPath(export_dir, kSavedModelVariablesDirectory);
  // Check for saver checkpoints in v2 format. Models exported in the checkpoint
  // v2 format will have a variables.index file. The corresponding
  // variables are stored in the variables.data-?????-of-????? files.
  const string variables_index_path = tensorflow::io::JoinPath(variables_directory, tensorflow::MetaFilename(kSavedModelVariablesFilename));
  if (!tensorflow::Env::Default()->FileExists(variables_index_path).ok()) 
  {
    LOG(INFO) << "The specified SavedModel has no variables; no checkpoints "
      "were restored.";
    return Status::OK();
  }
  const string variables_path = tensorflow::io::JoinPath(variables_directory, kSavedModelVariablesFilename);

  // Add variables to the graph.
  Tensor variables_path_tensor(DataType::DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = variables_path;

  std::vector<std::pair<string, Tensor>> inputs = { { variable_filename_const_op_name.ToString(), variables_path_tensor } };

  AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return session->Run(run_options, inputs, {}, { restore_op_name.ToString() },
                      nullptr /* outputs */, &run_metadata);
}

Status RunLegacyInitOp(const RunOptions& run_options, const string& export_dir,
                       const MetaGraphDef& meta_graph_def,
                       const std::vector<tensorflow::AssetFileDef>& asset_file_defs,
                       Session* session) 
{
  LOG(INFO) << "Running LegacyInitOp on SavedModel bundle.";
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto init_op_it = collection_def_map.find(kSavedModelLegacyInitOpKey);
  if (init_op_it != collection_def_map.end()) 
  {
    if (init_op_it->second.node_list().value_size() != 1)
      return FailedPrecondition(tensorflow::strings::StrCat("Expected exactly one serving init op in : ", export_dir));

    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    const tensorflow::StringPiece legacy_init_op_name = init_op_it->second.node_list().value(0);
    return session->Run(run_options, inputs, {},
                        { legacy_init_op_name.ToString() }, nullptr /* outputs */,
                        &run_metadata);
  }
  return Status::OK();
}

Status GetAssetFileDefs(const MetaGraphDef& meta_graph_def,
                        std::vector<tensorflow::AssetFileDef>* asset_file_defs) 
{
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto assets_it = collection_def_map.find(kSavedModelAssetsKey);
  if (assets_it == collection_def_map.end()) 
    return Status::OK();

  const auto& any_assets = assets_it->second.any_list().value();
  for (const auto& any_asset : any_assets) 
  {
    tensorflow::AssetFileDef asset_file_def;
    ParseAny(any_asset, &asset_file_def, "tensorflow.AssetFileDef");
    asset_file_defs->push_back(asset_file_def);
  }
  return Status::OK();
}

Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundle* const bundle,
                      const char* device) 
{
  if (!tensorflow::MaybeSavedModelDirectory(export_dir)) {
    return Status(Code::NOT_FOUND,
      "SavedModel not found in export directory: " + export_dir);
  }
  LOG(INFO) << "Loading SavedModel from: " << export_dir;

  SavedModel saved_model_proto;
  ReadSavedModel(export_dir, &saved_model_proto);

  FindMetaGraphDefToLoad(saved_model_proto, tags, &bundle->meta_graph_def);

  GraphDef* GDef = bundle->meta_graph_def.mutable_graph_def();
  for (int i = 0; i < GDef->node_size(); i++) 
  {
      auto Node = GDef->mutable_node(i);
      if (Node->device().empty())
        Node->set_device(device);
  }

  LoadMetaGraphIntoSession(bundle->meta_graph_def, session_options, &bundle->session);

  std::vector<tensorflow::AssetFileDef> asset_file_defs;
  GetAssetFileDefs(bundle->meta_graph_def, &asset_file_defs);
  RunRestore(run_options, export_dir,
              bundle->meta_graph_def.saver_def().restore_op_name(),
              bundle->meta_graph_def.saver_def().filename_tensor_name(),
              asset_file_defs, bundle->session.get());
  if (HasMainOp(bundle->meta_graph_def)) 
  {
    RunMainOp(run_options, export_dir, bundle->meta_graph_def, asset_file_defs, bundle->session.get());
  }
  else 
  {
    RunLegacyInitOp(run_options, export_dir, bundle->meta_graph_def, asset_file_defs, bundle->session.get());
  }
  return Status::OK();
}

TF_Operation* ToOperation(Node* node) 
{
    return static_cast<TF_Operation*>(static_cast<void*>(node));
}

static void GraphImportGraphDefLocked(TF_Graph* graph, const GraphDef& def,
                                      const TF_ImportGraphDefOptions* opts,
                                      TF_ImportGraphDefResults* tf_results,
                                      TF_Status* status)
                                      EXCLUSIVE_LOCKS_REQUIRED(graph->mu) 
{
  const int last_node_id = graph->graph.num_node_ids();
  tensorflow::ImportGraphDefResults results;
  status->status = tensorflow::ImportGraphDef(opts->opts, def, &graph->graph,
    &graph->refiner, &results);
  if (!status->status.ok()) return;

  // Add new nodes to name_map
  for (int i = last_node_id; i < graph->graph.num_node_ids(); ++i) {
    auto* node = graph->graph.FindNodeId(i);
    if (node != nullptr) graph->name_map[node->name()] = node;
  }

  // Populate return_tensors
  DCHECK(tf_results->return_tensors.empty());
  tf_results->return_tensors.resize(results.return_tensors.size());
  for (int i = 0; i < results.return_tensors.size(); ++i) {
    tf_results->return_tensors[i].oper =
      ToOperation(results.return_tensors[i].first);
    tf_results->return_tensors[i].index = results.return_tensors[i].second;
  }

  // Populate return_nodes
  DCHECK(tf_results->return_nodes.empty());
  tf_results->return_nodes.resize(results.return_nodes.size());
  for (int i = 0; i < results.return_nodes.size(); ++i) {
    tf_results->return_nodes[i] = ToOperation(results.return_nodes[i]);
  }

  // Populate missing unused map keys
  DCHECK(tf_results->missing_unused_key_names.empty());
  DCHECK(tf_results->missing_unused_key_indexes.empty());
  DCHECK(tf_results->missing_unused_key_names_data.empty());

  size_t size = results.missing_unused_input_map_keys.size();
  tf_results->missing_unused_key_names.resize(size);
  tf_results->missing_unused_key_indexes.resize(size);

  for (int i = 0; i < size; ++i) {
    TensorId id = results.missing_unused_input_map_keys[i];
    tf_results->missing_unused_key_names_data.push_back(id.first.ToString());
    tf_results->missing_unused_key_names[i] =
      tf_results->missing_unused_key_names_data.back().c_str();
    tf_results->missing_unused_key_indexes[i] = id.second;
  }
}

__declspec(dllexport) TF_Session* __stdcall TF_LoadSessionFromSavedModelOnDevice(const TF_SessionOptions* session_options, const TF_Buffer* run_options,
                                                                                 const char* export_dir, const char* const* tags, int tags_len,
                                                                                 TF_Graph* graph, const char* device, TF_Status* status) 
{
    if (!graph->name_map.empty()) 
    {
        status->status = InvalidArgument("Graph is non-empty.");
        return nullptr;
    }

    {
        const char* debug_allocator_str = std::getenv("TF_GPU_ALLOCATOR");
        if (debug_allocator_str != nullptr &&
            strcmp(debug_allocator_str, "cuda_malloc") == 0)
            LOG(INFO) << "cuda_malloc";
        else
            LOG(INFO) << "not cuda_malloc";
    }

    RunOptions run_options_proto;
    if (run_options != nullptr && !run_options_proto.ParseFromArray(run_options->data, run_options->length)) 
    {
        status->status = InvalidArgument("Unparseable RunOptions proto");
        return nullptr;
    }

    std::unordered_set<string> tag_set;
    for (int i = 0; i < tags_len; i++)
        tag_set.insert(string(tags[i]));

    GraphDef GDNew;

    tensorflow::SavedModelBundle bundle;
    status->status = LoadSavedModel(session_options->options, run_options_proto, export_dir, tag_set, &bundle, device);
    if (!status->status.ok()) return nullptr;

    // Create a TF_Graph from the MetaGraphDef. This is safe as long as Session
    // extends using GraphDefs. The Graph instance is different, but equivalent
    // to the one used to create the session.
    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefResults results;
    GraphImportGraphDefLocked(graph, bundle.meta_graph_def.graph_def(), import_opts, &results, status);
    TF_DeleteImportGraphDefOptions(import_opts);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    
    TF_Session* session = new TF_Session(bundle.session.release(), graph);

    /*GraphDef GDef = bundle.meta_graph_def.graph_def();
    for (int i = 0; i < GDef.node_size(); i++)
    {
        auto Node = GDef.node(i);
        auto Attributes = Node.attr();
        if (Attributes.count("is_training") > 0)
        {
            LOG(INFO) << Attributes["is_training"].b();
        }
    }*/

    //graph->sessions[session] = Status::OK();
    session->last_num_graph_nodes = graph->graph.num_node_ids();
    return session;
}

__declspec(dllexport) void __stdcall TF_FreeAllMemory()
{
	tensorflow::GPUProcessState::singleton()->~GPUProcessState();
}